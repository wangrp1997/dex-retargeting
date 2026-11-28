"""Exercise the installed SDK through a virtual serial port, never /dev/dexhand."""
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
from threading import Event, Thread
from time import sleep

import pytest


@pytest.fixture
def tracer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'example/vector_retargeting'))
    import gs_s1_io_trace
    return gs_s1_io_trace


@pytest.fixture
def serial_client(tmp_path, tracer):
    if not shutil.which('g++') or not tracer.SDK_LIBRARY.exists():
        pytest.skip('Installed Linux SDK and compiler required')
    source = tmp_path / 'client.cpp'
    source.write_text(r'''
#include <dexhand_driver/impl/rs485.hpp>
#include <string>
extern "C" int run(const char* path) {
    if (std::string(path).rfind("/dev/pts/", 0) != 0) return 90;
    dexhand_driver::SerialDriver driver(path, 0, 1000000, 40);
    if (!driver.connect()) return 91;
    const bool a = driver.execCmdPing();
    const bool b = driver.execCmdCtlDofs({1,0,0}, {0,0,40,0,80,0});
    const bool c = driver.execCmdCtlDofs({1,0,0}, {0,0,40,0,80,0});
    const bool d = driver.execCmdPing();
    return a && b && !c && !d ? 0 : 92;
}
''')
    # Load the SDK through RTLD_LOCAL, as pluginlib does in ros2_control.
    loader = tmp_path / 'loader.cpp'
    loader.write_text(r'''
#include <dlfcn.h>
#include <cstdio>
int main(int argc, char** argv) {
    if (argc != 3) return 93;
    void* handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!handle) { std::fprintf(stderr, "%s\n", dlerror()); return 94; }
    auto run = reinterpret_cast<int(*)(const char*)>(dlsym(handle, "run"));
    const int result = run ? run(argv[2]) : 95;
    dlclose(handle);
    return result;
}
''')
    client = tmp_path / 'libclient.so'
    executable = tmp_path / 'loader'
    subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', str(source), '-L/usr/local/lib',
                    '-Wl,-rpath,/usr/local/lib', '-ldexhand_driver', '-o', str(client)], check=True, capture_output=True)
    subprocess.run(['g++', str(loader), '-ldl', '-o', str(executable)], check=True, capture_output=True)
    return executable, client


def emulate(executable, client, env):
    master, slave = os.openpty()
    path = os.ttyname(slave)
    assert path.startswith('/dev/pts/')
    stop = Event()
    packets, errors = [], []

    def server():
        pending = bytearray()
        try:
            while not stop.is_set():
                if not select.select([master], [], [], .02)[0]:
                    continue
                pending.extend(os.read(master, 4096))
                while len(pending) >= 7:
                    assert pending[:2] == b'\x4c\xaa'
                    size = int.from_bytes(pending[4:6], 'little')
                    if len(pending) < size + 7:
                        break
                    packet = bytes(pending[:size+7]); del pending[:size+7]
                    assert sum(packet[:-1]) & 255 == packet[-1]
                    packets.append(packet)
                    if len(packets) >= 4:
                        continue  # An intentional timeout, after a valid NACK.
                    sleep(.02)
                    payload = b'\xff' if len(packets) == 3 else b'\x00'
                    frame = bytes([0xaa, 0x4c, packet[2], packet[3], 1, 0]) + payload
                    os.write(master, frame + bytes([sum(frame) & 255]))
        except Exception as error:
            errors.append(error)

    thread = Thread(target=server, daemon=True)
    thread.start()
    try:
        result = subprocess.run([str(executable), str(client), path], env=env,
                                 capture_output=True, text=True, timeout=10.)
        assert result.returncode == 0, result.stderr
        assert not errors, errors
        return packets
    finally:
        stop.set()
        thread.join(timeout=1.)
        os.close(slave); os.close(master)


def test_serial_trace_preserves_packets_and_distinguishes_nack_and_timeout(tracer, serial_client, tmp_path):
    library, sdk = tracer.build_trace_library(tmp_path)
    executable, client = serial_client
    baseline = emulate(executable, client, dict(os.environ))
    trace_file = tmp_path / 'io_trace.jsonl'
    env = dict(os.environ, LD_PRELOAD=str(library), BOTYARD_IO_TRACE_FILE=str(trace_file),
               BOTYARD_IO_TRACE_SDK=str(sdk))
    traced = emulate(executable, client, env)
    assert traced == baseline
    rows = [json.loads(line) for line in trace_file.read_text().splitlines()]
    sends = [row for row in rows if row['event'] == 'serial_send']
    responses = [row for row in rows if row['event'] == 'serial_response']
    acks = [row for row in rows if row['event'] == 'serial_ack']
    assert len(sends) == len(responses) == len(acks) == 4
    assert [row['cmd_id'] for row in sends] == [0, 0xa0, 0xa0, 0]
    assert [row['success'] for row in responses] == [True, True, True, False]
    assert [row['success'] for row in acks] == [True, True, False, False]
    assert len({row['request_id'] for row in sends}) == 4
    for send, response, ack in zip(sends, responses, acks):
        assert send['request_id'] == response['request_id'] == ack['request_id']
        assert send['ack_context_id'] == response['ack_context_id'] == ack['ack_context_id']
        assert ack['begin_steady_ns'] <= send['begin_steady_ns'] <= send['end_steady_ns']
        assert send['end_steady_ns'] <= response['begin_steady_ns'] <= response['end_steady_ns'] <= ack['end_steady_ns']
    summary = tracer.summarize_io(trace_file)
    assert summary['clean_shutdown']
    assert summary['dropped_events'] == 0
    assert summary['send_failures'] == 0
    assert summary['response_failures'] == 1
    assert summary['ack_failures'] == 2


def test_missing_trace_is_unknown_not_zero_latency(tracer, tmp_path):
    summary = tracer.summarize_io(tmp_path / 'absent.jsonl')
    assert summary['available'] is False
