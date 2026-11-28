"""Build and inspect opt-in Linux RS485 SDK diagnostics without replacing installed libraries."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4


SYMBOLS = (
    '_ZN14dexhand_driver12SerialDriver11sendCommandEhRKSt6vectorIhSaIhEEh',
    '_ZN14dexhand_driver12SerialDriver12waitResponseEhRSt6vectorIhSaIhEE',
    '_ZN14dexhand_driver12SerialDriver22sendCommandAndCheckAckEhRKSt6vectorIhSaIhEEh',
)
SDK_LIBRARY = Path('/usr/local/lib/libdexhand_driver.so')


def build_trace_library(root, *, sdk_library=SDK_LIBRARY):
    compiler = shutil.which('g++')
    if compiler is None:
        raise RuntimeError('hardware_io_trace requires g++')
    sdk_library = Path(sdk_library).resolve(strict=True)
    symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(sdk_library)], text=True)
    if any(symbol not in symbols.split() for symbol in SYMBOLS):
        raise RuntimeError('Installed SDK does not export the required RS485 tracing functions')
    source = Path(__file__).with_suffix('.cpp')
    digest = hashlib.sha256(source.read_bytes() + sdk_library.read_bytes()).hexdigest()[:16]
    cache = Path(root).expanduser().resolve() / '.io-trace-cache' / digest
    library = cache / 'libbotyard_io_trace.so'
    if not library.is_file():
        from tempfile import TemporaryDirectory
        cache.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / library.name
            result = subprocess.run([compiler, '-std=c++17', '-O2', '-shared', '-fPIC', '-pthread',
                                     str(source), '-ldl', '-o', str(output)], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('Could not build hardware IO tracer:\n' + result.stderr)
            output.replace(library)
    return library, sdk_library


def prepare_trace(root):
    root = Path(root).expanduser().resolve()
    library, sdk = build_trace_library(root)
    directory = root / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8])
    directory.mkdir(parents=True)
    return directory, library, sdk


def summarize_io(path):
    """SDK send/return and validated ACK timings are not motor application times."""
    from gs_s1_hardware_log import _stats
    path = Path(path)
    if not path.is_file():
        return dict(available=False, path=str(path))
    sends, responses, acks, footers = [], [], [], []
    invalid = 0
    for line in path.open():
        try:
            row = json.loads(line)
        except ValueError:
            invalid += 1
            continue
        kind = row.get('event')
        if kind == 'serial_send': sends.append(row)
        elif kind == 'serial_response': responses.append(row)
        elif kind == 'serial_ack': acks.append(row)
        elif kind == 'io_trace_end': footers.append(row)
    def durations(rows):
        return [(r['end_steady_ns'] - r['begin_steady_ns']) / 1e6 for r in rows]

    # Unsupported startup/version probes must not be mistaken for motion-time
    # communication failures. Keep separate statistics for every command ID.
    per_command = {}
    for cmd in sorted({r['cmd_id'] for r in sends + responses + acks}):
        sent = [r for r in sends if r['cmd_id'] == cmd]
        replied = [r for r in responses if r['cmd_id'] == cmd]
        checked = [r for r in acks if r['cmd_id'] == cmd]
        per_command[f'0x{cmd:02x}'] = dict(
            sends=len(sent), response_failures=sum(not r['success'] for r in replied),
            ack_failures=sum(not r['success'] for r in checked),
            send_ms=_stats(durations(sent)), response_wait_ms=_stats(durations(replied)),
            ack_transaction_ms=_stats(durations(checked)))
    return dict(available=True, path=str(path), send_calls=len(sends), response_calls=len(responses),
                ack_calls=len(acks), send_failures=sum(not r['success'] for r in sends),
                response_failures=sum(not r['success'] for r in responses),
                ack_failures=sum(not r['success'] for r in acks),
                send_ms=_stats(durations(sends)), response_wait_ms=_stats(durations(responses)),
                ack_transaction_ms=_stats(durations(acks)), clean_shutdown=bool(footers),
                dropped_events=sum(r['dropped_events'] for r in footers) if footers else None,
                invalid_json_lines=invalid, commands=per_command,
                note='Send timestamps bracket the SDK serial send function; response timestamps '
                     'bracket waiting/parsing. ACK success means protocol acceptance, not motion completion. '
                     'Host timestamps do not observe bytes on the electrical bus or firmware application.')
