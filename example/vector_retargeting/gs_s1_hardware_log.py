"""Optional asynchronous hardware diagnostics and offline summaries; no motion publisher."""
import argparse
from collections import Counter, deque
from datetime import datetime
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic, monotonic_ns
from uuid import uuid4

from gs_s1_hardware_commands import FINGER_JOINTS


def _json_value(value):
    # Invalid hardware samples can contain NaNs. Preserve missing values as null.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


class HardwareSessionLog:
    """Bound disk I/O outside the relay callbacks and account for any lost events."""

    def __init__(self, root, metadata, *, queue_size=4096):
        self.directory = Path(root).expanduser().resolve() / (
            datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8])
        self.directory.mkdir(parents=True)
        self.metadata = dict(format_version=1, created_at=datetime.now().astimezone().isoformat(),
                             **metadata)
        self._metadata_path = self.directory / 'session.json'
        self._metadata_path.write_text(json.dumps(self.metadata, indent=2) + '\n')
        self.queue, self.stop = Queue(maxsize=queue_size), Event()
        self.dropped = self.written = self.sequence = 0
        self.error = None
        self.closed = False
        self.thread = Thread(target=self._write, name='botyard-hardware-log', daemon=True)
        self.thread.start()

    def record(self, event, ros_time_ns, **fields):
        if self.closed:
            return
        self.sequence += 1
        row = dict(event=event, event_id=self.sequence, ros_time_ns=int(ros_time_ns),
                   steady_time_ns=monotonic_ns(), **fields)
        if self.error is not None:
            self.dropped += 1
            return
        try:
            self.queue.put_nowait(row)
        except Full:
            self.dropped += 1

    def _write(self):
        try:
            with (self.directory / 'events.jsonl').open('w', buffering=65536) as output:
                last_flush = monotonic()
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        row = self.queue.get(timeout=.1)
                    except Empty:
                        output.flush()
                        continue
                    output.write(json.dumps(_json_value(row), allow_nan=False, separators=(',', ':')) + '\n')
                    self.written += 1
                    if monotonic() - last_flush >= 1.:
                        output.flush()
                        last_flush = monotonic()
        except Exception as error:
            self.error = str(error)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.thread.join(timeout=5.)
        writer_alive = self.thread.is_alive()
        self.metadata.update(closed_at=datetime.now().astimezone().isoformat(),
                             attempted_events=self.sequence, written_events=self.written,
                             dropped_events=self.dropped, writer_error=self.error,
                             pending_events=self.queue.qsize(), writer_stopped=not writer_alive)
        self.metadata['unwritten_events'] = self.sequence - self.written
        self._metadata_path.write_text(json.dumps(self.metadata, indent=2) + '\n')
        if not writer_alive and (self.directory / 'events.jsonl').exists():
            summary = summarize(self.directory)
            (self.directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def _stats(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return dict(samples=0, mean=None, p50=None, p95=None, maximum=None)

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (position - lo) * (values[hi] - values[lo])
    return dict(samples=len(values), mean=sum(values) / len(values), p50=percentile(.5),
                p95=percentile(.95), maximum=values[-1])


def summarize(directory):
    """Describe observed rates and source age without claiming a motor delay estimate."""
    directory = Path(directory)
    counts, first, last = Counter(), {}, {}
    ages, poll_ms, read_ms, solve_ms, detect_ms, mano_ms, serial_ms = [], [], [], [], [], [], []
    target_command_errors = []
    broadcast_errors, acquisition_errors = [], []
    commands = deque(maxlen=256)
    configured_rates = set()
    invalid_lines = invalid_samples = duplicate_samples = 0
    previous_sample = None
    sample_time_discontinuities = 0
    first_sample_ns = last_sample_ns = None
    fresh_samples = pending_commands = 0
    for line in (directory / 'events.jsonl').open():
        try:
            row = json.loads(line)
            kind, stamp = row['event'], row['steady_time_ns']
        except (ValueError, KeyError, TypeError):
            invalid_lines += 1
            continue
        counts[kind] += 1
        first.setdefault(kind, stamp)
        last[kind] = stamp
        if kind == 'target' and row.get('accepted'):
            ages.append(row['source_age_s'] * 1000)
        elif kind == 'command':
            commands.append((row['ros_time_ns'], dict(zip(row['joint_names'], row['positions']))))
            errors = row.get('target_error_rad', [])
            if errors:
                target_command_errors.append(math.degrees(math.sqrt(sum(e*e for e in errors) / len(errors))))
            pending_commands += bool(row.get('ramp_pending_joints'))
        elif kind == 'feedback':
            error = _feedback_error(commands, row)
            if error is not None:
                broadcast_errors.append(error)
        elif kind == 'pipeline':
            timing = row['timing']
            if timing.get('kind') == 'observation' and timing.get('tracking'):
                detect_ms.append(timing['detection_s'] * 1000)
                if timing.get('mano_fit_s') is not None:
                    mano_ms.append(timing['mano_fit_s'] * 1000)
            elif timing.get('kind') == 'solution':
                if timing.get('solve_s') is not None:
                    solve_ms.append(timing['solve_s'] * 1000)
                if timing.get('serial_mano_and_solve_s') is not None:
                    serial_ms.append(timing['serial_mano_and_solve_s'] * 1000)
        elif kind == 'hardware_sample':
            if not row.get('valid'):
                invalid_samples += 1
                continue
            key = row['sequence']
            if key == previous_sample:
                duplicate_samples += 1
                continue
            previous_sample = key
            fresh_samples += 1
            if first_sample_ns is None:
                first_sample_ns = row['source_stamp_ns']
            if last_sample_ns is not None and row['source_stamp_ns'] <= last_sample_ns:
                sample_time_discontinuities += 1
            last_sample_ns = row['source_stamp_ns']
            if row.get('configured_io_rate_hz') is not None:
                configured_rates.add(row['configured_io_rate_hz'])
            if row.get('poll_interval_s') is not None and row['poll_interval_s'] > 0:
                poll_ms.append(row['poll_interval_s'] * 1000)
            if row.get('read_duration_s') is not None:
                read_ms.append(row['read_duration_s'] * 1000)
            error = _feedback_error(commands, row)
            if error is not None:
                acquisition_errors.append(error)
    rates = {kind: ((count - 1) * 1e9 / (last[kind] - first[kind])
                    if count > 1 and last[kind] > first[kind] else None)
             for kind, count in counts.items()}
    fresh_hz = ((fresh_samples - 1) * 1e9 / (last_sample_ns - first_sample_ns)
                if fresh_samples > 1 and last_sample_ns > first_sample_ns and not sample_time_discontinuities else None)
    return dict(event_counts=dict(counts), callback_rates_hz=rates,
                target_source_age_ms=_stats(ages), detection_ms=_stats(detect_ms),
                mano_fit_ms=_stats(mano_ms), solve_ms=_stats(solve_ms),
                serial_mano_and_solve_ms=_stats(serial_ms),
                target_to_command_rms_deg=_stats(target_command_errors),
                broadcast_command_to_feedback_rms_deg=_stats(broadcast_errors),
                acquisition_command_to_feedback_rms_deg=_stats(acquisition_errors),
                commands_still_approaching_target=pending_commands,
                configured_io_rates_hz=sorted(configured_rates), fresh_hardware_samples=fresh_samples,
                fresh_hardware_sample_hz=fresh_hz, invalid_hardware_samples=invalid_samples,
                duplicate_hardware_samples=duplicate_samples, io_poll_interval_ms=_stats(poll_ms),
                hardware_stamp_discontinuities=sample_time_discontinuities,
                io_read_duration_ms=_stats(read_ms), invalid_json_lines=invalid_lines,
                note='Callback rates are receipt rates. joint_states can repeat old hardware readings; '
                     'only valid, distinct HandSample acquisitions count as fresh hardware samples. '
                     'Target source age includes capture, vision, solve and delivery. Command events '
                     'mark relay publication, not RS485 application or physical motion onset. '
                     'A motor tracking delay is not inferred from these summary statistics.')


def _feedback_error(commands, row):
    # Compare with the last command already published at the feedback's source
    # time, never with a newer command received during transport/readout.
    command = next((values for stamp, values in reversed(commands) if stamp <= row['source_stamp_ns']), None)
    if command is None:
        return None
    values = dict(zip(row['joint_names'], row['positions']))
    errors = [(values[n] - command[n])**2 for n in FINGER_JOINTS
              if n in values and n in command and values[n] is not None and command[n] is not None]
    return math.degrees(math.sqrt(sum(errors) / len(errors))) if errors else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Hardware log session to summarize offline')
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))


if __name__ == '__main__':
    main()
