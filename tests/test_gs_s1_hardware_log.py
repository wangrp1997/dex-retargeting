"""Diagnostic recording must be optional, bounded and honest about sample freshness."""
import json
import math
from pathlib import Path
from threading import Event

import pytest


@pytest.fixture
def logs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'example/vector_retargeting'))
    import gs_s1_hardware_log
    return gs_s1_hardware_log


def test_log_drains_on_close_and_preserves_missing_samples(logs, tmp_path):
    log = logs.HardwareSessionLog(tmp_path, {'hand': 'left'})
    log.record('feedback', 100, source_stamp_ns=90, joint_names=['FFJ2'], positions=[float('nan')])
    log.record('target', 110, accepted=True, source_age_s=.15)
    log.close()
    log.close()
    assert not log.thread.is_alive()
    rows = [json.loads(line) for line in (log.directory / 'events.jsonl').read_text().splitlines()]
    assert [row['event_id'] for row in rows] == [1, 2]
    assert rows[0]['positions'] == [None]
    metadata = json.loads((log.directory / 'session.json').read_text())
    assert metadata['written_events'] == 2
    assert metadata['unwritten_events'] == metadata['dropped_events'] == 0
    summary = json.loads((log.directory / 'summary.json').read_text())
    assert summary['target_source_age_ms']['mean'] == pytest.approx(150.)
    assert summary['fresh_hardware_sample_hz'] is None


def test_queue_overflow_is_counted_and_never_blocks_callbacks(logs, tmp_path, monkeypatch):
    release = Event()
    original = logs.HardwareSessionLog._write

    def delayed_writer(self):
        release.wait(timeout=5.)
        original(self)

    monkeypatch.setattr(logs.HardwareSessionLog, '_write', delayed_writer)
    log = logs.HardwareSessionLog(tmp_path, {}, queue_size=1)
    try:
        log.record('controller_status', 1, active=False)
        log.record('controller_status', 2, active=True)
        assert log.dropped == 1
    finally:
        release.set()
        log.close()
    metadata = json.loads((log.directory / 'session.json').read_text())
    assert metadata['written_events'] == metadata['unwritten_events'] == metadata['dropped_events'] == 1


def test_writer_failure_is_reported_in_metadata(logs, tmp_path, monkeypatch):
    monkeypatch.setattr(logs.HardwareSessionLog, '_write', lambda self: setattr(self, 'error', 'injected disk failure'))
    log = logs.HardwareSessionLog(tmp_path, {})
    log.thread.join(timeout=1.)
    log.record('controller_status', 1, active=True)
    log.close()
    metadata = json.loads((log.directory / 'session.json').read_text())
    assert metadata['writer_error'] == 'injected disk failure'
    assert metadata['dropped_events'] == metadata['unwritten_events'] == 1


def test_summary_distinguishes_republished_feedback_and_real_acquisitions(logs, tmp_path):
    rows = []
    for k in range(100):
        rows.append(dict(event='feedback', steady_time_ns=k * 10_000_000,
                         source_stamp_ns=k * 10_000_000, joint_names=[], positions=[]))
    for k in range(10):
        sample = dict(event='hardware_sample', steady_time_ns=k * 100_000_000,
                      source_stamp_ns=k * 100_000_000, sequence=k, valid=True,
                      configured_io_rate_hz=10., poll_interval_s=.1, read_duration_s=.005,
                      joint_names=[], positions=[])
        rows.append(sample)
        # A newer publication timestamp is not a new acquisition sequence.
        rows.append(dict(sample, steady_time_ns=k * 100_000_000 + 1, source_stamp_ns=k * 100_000_000 + 1))
    rows.sort(key=lambda row: row['steady_time_ns'])
    (tmp_path / 'events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    summary = logs.summarize(tmp_path)
    assert summary['callback_rates_hz']['feedback'] == pytest.approx(100.)
    assert summary['fresh_hardware_sample_hz'] == pytest.approx(10.)
    assert summary['fresh_hardware_samples'] == summary['duplicate_hardware_samples'] == 10
    assert summary['io_poll_interval_ms']['mean'] == pytest.approx(100.)


def test_feedback_is_compared_to_command_at_acquisition_time(logs, tmp_path):
    rows = [dict(event='command', steady_time_ns=10, ros_time_ns=10,
                 joint_names=['FFJ2'], positions=[.1], target_error_rad=[.2], ramp_pending_joints=['FFJ2']),
            dict(event='command', steady_time_ns=30, ros_time_ns=30,
                 joint_names=['FFJ2'], positions=[.5], target_error_rad=[0.], ramp_pending_joints=[]),
            dict(event='hardware_sample', steady_time_ns=40, source_stamp_ns=20,
                 sequence=1, valid=True, configured_io_rate_hz=10., joint_names=['FFJ2'], positions=[.12])]
    (tmp_path / 'events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows) + '{unfinished')
    summary = logs.summarize(tmp_path)
    assert summary['acquisition_command_to_feedback_rms_deg']['mean'] == pytest.approx(math.degrees(.02))
    assert summary['commands_still_approaching_target'] == 1
    assert summary['invalid_json_lines'] == 1


def test_serial_and_latest_timings_are_kept_separate(logs, tmp_path):
    timings = [dict(kind='observation', tracking=True, detection_s=.01, mano_fit_s=None),
               dict(kind='solution', serial_mano_and_solve_s=.14),
               dict(kind='observation', tracking=True, detection_s=.012, mano_fit_s=.007),
               dict(kind='solution', solve_s=.1)]
    (tmp_path / 'events.jsonl').write_text(''.join(json.dumps(dict(
        event='pipeline', steady_time_ns=k * 100_000_000, timing=timing)) + '\n'
        for k, timing in enumerate(timings)))
    summary = logs.summarize(tmp_path)
    assert summary['solve_ms']['mean'] == pytest.approx(100.)
    assert summary['serial_mano_and_solve_ms']['mean'] == pytest.approx(140.)
    assert summary['mano_fit_ms']['samples'] == 1
