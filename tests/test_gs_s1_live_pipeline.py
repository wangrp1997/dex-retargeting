"""No camera, ROS graph or hardware required to exercise live scheduling."""
from pathlib import Path
from queue import Queue
from time import monotonic, sleep
from types import SimpleNamespace
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'example/vector_retargeting'))
from gs_s1_live_pipeline import LatestCamera, LiveRetargeting, PoseResult, SolverProcess


def eventually(callback, timeout=10.):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        result = callback()
        if result:
            return result
        sleep(.01)
    raise AssertionError('Timed out waiting for worker')


class FakeCapture:
    def __init__(self):
        self.frames = Queue()
        self.released = False

    def read(self):
        frame = self.frames.get(timeout=5.)
        return frame is not None, frame

    def release(self):
        self.released = True
        self.frames.put(None)


def test_camera_drains_old_frames_and_shutdown_unblocks_read():
    capture = FakeCapture()
    reader = LatestCamera(capture)
    try:
        for k in range(20):
            capture.frames.put(np.full((2, 2, 3), k))
        eventually(lambda: reader.latest is not None and reader.latest.sequence == 19)
        latest = reader.take()
        assert latest.sequence == 19
        assert (latest.image == 19).all()
        assert monotonic() - latest.captured_at < 1.
        assert reader.take(timeout=0.) is None
    finally:
        reader.close()
    assert capture.released
    assert not reader.thread.is_alive()


def test_camera_failure_is_reported():
    capture = FakeCapture()
    reader = LatestCamera(capture)
    capture.frames.put(None)
    try:
        with pytest.raises(RuntimeError, match='stopped returning frames'):
            reader.take(timeout=1.)
    finally:
        reader.close()


class FakeWorker:
    ready = True

    def __init__(self):
        self.tasks, self.results = [], []

    def send(self, task):
        self.tasks.append(task)

    def poll(self):
        return self.results.pop(0) if self.results else None

    def finish(self):
        task = self.tasks[-1]
        self.results.append(PoseResult(task.sequence, task.generation, task.captured_at,
                                       task.previous + .1, task.previous + .1, .12))


def submit(lane, sequence, timestamp):
    return lane.submit(sequence, timestamp, np.zeros((21, 3)), np.zeros((778, 3)), timestamp)


def test_busy_solver_drops_inputs_then_uses_latest_and_publishes_once():
    worker = FakeWorker()
    lane = LiveRetargeting(worker, np.zeros(16))
    lane.observe(True, 1.)
    assert submit(lane, 0, 1.)
    for k in range(1, 10):
        lane.observe(True, 1. + k * .01)
        assert not submit(lane, k, 1. + k * .01)
    assert len(worker.tasks) == 1
    worker.finish()
    result = lane.accept(1.12)
    assert result.captured_at == 1.
    assert lane.accept(1.12) is None
    assert submit(lane, 10, 1.14)
    assert len(worker.tasks) == 2
    assert worker.tasks[-1].sequence == 10
    assert worker.tasks[-1].dt == .1
    np.testing.assert_array_equal(worker.tasks[-1].previous, .1)


@pytest.mark.parametrize('reason', ['loss', 'reacquired', 'expired', 'capture_gap'])
def test_stale_results_do_not_advance_actual_command(reason):
    worker = FakeWorker()
    lane = LiveRetargeting(worker, np.zeros(16))
    lane.observe(True, 1.)
    submit(lane, 0, 1.)
    worker.finish()
    now = 1.2
    if reason in ('loss', 'reacquired'):
        lane.observe(False, 1.1)
        if reason == 'reacquired':
            lane.observe(True, 1.15)
    else:
        now = 1.6
        if reason == 'capture_gap':
            lane.observe(True, now)
            # Even a fresh reply from the earlier epoch must be rejected.
            lane.max_age = 1.
    assert lane.accept(now) is None
    assert lane.discarded == 1
    assert lane.accepted_at is None
    np.testing.assert_array_equal(lane.previous, 0.)
    lane.observe(True, now + .01)
    assert submit(lane, 2, now + .01)
    np.testing.assert_array_equal(worker.tasks[-1].previous, 0.)


def test_dispatch_waits_for_readiness_and_rejects_expired_observation():
    worker = FakeWorker()
    worker.ready = False
    lane = LiveRetargeting(worker, np.zeros(16))
    lane.observe(True, 1.)
    assert not submit(lane, 0, 1.)
    worker.ready = True
    assert not lane.submit(0, 1., np.zeros((21, 3)), np.zeros((778, 3)), 1.6)
    assert not lane.submit(0, 1., np.zeros((21, 3)), np.zeros((778, 3)), .9)


def test_nonfinite_reply_cannot_be_published():
    worker = FakeWorker()
    lane = LiveRetargeting(worker, np.zeros(16))
    lane.observe(True, 1.)
    submit(lane, 0, 1.)
    worker.finish()
    worker.results[0].q[:] = np.nan
    with pytest.raises(ValueError, match='Invalid retargeted'):
        lane.accept(1.1)
    np.testing.assert_array_equal(lane.previous, 0.)


class SlowRetargeter:
    def __init__(self, config):
        self.delay = .08 if config is None else config
        self.reference = SimpleNamespace(last=np.zeros(16))
        self.output = SimpleNamespace(last=np.zeros(16))
        self.protected_closure = SimpleNamespace(solver=SimpleNamespace(last=np.zeros(16)))

    def solve_mano(self, joints, vertices, timestamp, dt):
        # Check that all stages are reset to the actual command each time.
        np.testing.assert_array_equal(self.reference.last, self.output.last)
        np.testing.assert_array_equal(self.protected_closure.solver.last, self.output.last)
        sleep(self.delay)
        self.output.last += dt
        return self.output.last.copy()


def failing_factory(config):
    raise ValueError('injected worker failure')


def test_spawned_worker_resets_discarded_warm_start_and_closes():
    worker = SolverProcess(None, factory=SlowRetargeter)
    pid = worker.process.pid
    try:
        eventually(lambda: worker.poll() or worker.ready)
        lane = LiveRetargeting(worker, np.zeros(16))
        first = monotonic()
        lane.observe(True, first)
        assert submit(lane, 0, first)
        lane.observe(False, first + .01)
        eventually(lambda: lane.accept(monotonic()) or lane.discarded)
        now = monotonic()
        lane.observe(True, now)
        assert submit(lane, 1, now)
        reply = eventually(lambda: lane.accept(monotonic()))
        np.testing.assert_allclose(reply.actuators, 1 / 30)
        assert lane.accepted == 1
    finally:
        worker.close()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_spawned_worker_reports_initialization_failure():
    worker = SolverProcess(None, factory=failing_factory)
    try:
        with pytest.raises(RuntimeError, match='injected worker failure'):
            eventually(worker.poll)
    finally:
        worker.close()


def test_shutdown_terminates_worker_busy_in_native_solve():
    worker = SolverProcess(30., factory=SlowRetargeter)
    pid = worker.process.pid
    try:
        eventually(lambda: worker.poll() or worker.ready)
        lane = LiveRetargeting(worker, np.zeros(16))
        now = monotonic()
        lane.observe(True, now)
        assert submit(lane, 0, now)
    finally:
        worker.close()
    worker.close()  # Idempotent cleanup after an interrupted stream.
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
