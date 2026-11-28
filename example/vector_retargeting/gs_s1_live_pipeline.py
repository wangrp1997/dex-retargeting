"""Bounded live camera/solver scheduling, independent of ROS and GUI state."""
from dataclasses import dataclass
import multiprocessing as mp
from threading import Condition, Event, Thread
from time import monotonic
import traceback

import numpy as np


@dataclass
class CameraFrame:
    sequence: int
    captured_at: float
    image: np.ndarray


class LatestCamera:
    """Drain the device continuously; retain only the most recent image."""

    def __init__(self, capture):
        self.capture = capture
        self.condition, self.stop = Condition(), Event()
        self.latest = self.error = None
        self.thread = Thread(target=self._run, name='botyard-camera', daemon=True)
        self.thread.start()

    def _run(self):
        sequence = 0
        try:
            while not self.stop.is_set():
                ok, image = self.capture.read()
                captured_at = monotonic()  # Host receipt, not sensor exposure time.
                if not ok:
                    raise RuntimeError('Camera stopped returning frames.')
                with self.condition:
                    self.latest = CameraFrame(sequence, captured_at, image)
                    sequence += 1
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def take(self, timeout=.02):
        with self.condition:
            self.condition.wait_for(lambda: self.latest is not None or self.error is not None,
                                    timeout=timeout)
            if self.error is not None:
                raise RuntimeError(str(self.error)) from self.error
            result, self.latest = self.latest, None
            return result

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.)
        # A disconnected camera can block read(); release it to unblock shutdown.
        self.capture.release()
        self.thread.join(timeout=1.)
        if self.thread.is_alive():
            raise RuntimeError('Camera reader did not stop after device release')


@dataclass
class SolverConfig:
    assets: str
    hand: str
    template: str | None
    smoothing: str
    gradient: str
    shape_model: str
    opposition: str = 'v4_2'

    def __post_init__(self):
        if self.opposition not in ('v4_2', 'inward'):
            raise ValueError('Unknown Botyard opposition mode')
        if self.opposition == 'inward' and self.shape_model != 'v4_2':
            raise ValueError('Inward opposition requires the v4_2 shape objective')


@dataclass
class PoseTask:
    sequence: int
    generation: int
    captured_at: float
    joints: np.ndarray
    vertices: np.ndarray
    previous: np.ndarray
    dt: float


@dataclass
class PoseResult:
    sequence: int
    generation: int
    captured_at: float
    q: np.ndarray
    actuators: np.ndarray
    solve_seconds: float


def build_botyard_solver(config):
    # Imported only inside the spawned process. No ROS node, detector, GUI or
    # mutable MuJoCo state is shared with the camera/MANO process.
    import sys
    from pathlib import Path
    from ament_index_python.packages import get_package_share_directory
    from gs_s1_mujoco import load_hand
    from gs_s1_rviz import build_solver
    sys.path.append(str(Path(__file__).resolve().parent / 'data/gs_s1_hand_baselines'))
    from botyard_live import make_botyard_retargeter

    meshes = Path(get_package_share_directory('dexhand_description')) / 'meshes/gs_s1'
    solver = build_solver(hand=config.hand, objective='botyard')[0]
    return make_botyard_retargeter(solver, load_hand(Path(config.assets), meshes, config.hand),
                                   meshes, config.hand, Path(config.template) if config.template else None,
                                   config.smoothing, gradient=config.gradient, shape_model=config.shape_model,
                                   opposition=config.opposition)


def _solver_loop(connection, factory, config):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        retargeter = factory(config)
        connection.send(('ready', None))
        while True:
            task = connection.recv()
            if task is None:
                break
            # Discarded results must not become the next physical warm start.
            retargeter.reference.last = task.previous.copy()
            retargeter.output.last = task.previous.copy()
            if retargeter.protected_closure is not None:
                retargeter.protected_closure.solver.last = task.previous.copy()
            start = monotonic()
            q = retargeter.solve_mano(task.joints, task.vertices, task.captured_at, task.dt)
            connection.send(('result', PoseResult(
                task.sequence, task.generation, task.captured_at, q,
                retargeter.output.last.copy(), monotonic() - start)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


class SolverProcess:
    """One in-flight solve, with explicit readiness/error/shutdown handling."""

    def __init__(self, config, *, factory=build_botyard_solver):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_solver_loop, args=(child, factory, config),
                                       name='botyard-retargeting', daemon=True)
        self.ready = False
        self.closed = False
        self.process.start()
        child.close()

    def poll(self):
        try:
            while self.connection.poll():
                kind, value = self.connection.recv()
                if kind == 'error':
                    raise RuntimeError('Botyard solver process failed:\n' + value)
                if kind == 'ready':
                    self.ready = True
                elif kind == 'result':
                    return value
        except (EOFError, OSError) as error:
            raise RuntimeError('Botyard solver process disconnected') from error
        if not self.process.is_alive():
            raise RuntimeError(f'Botyard solver process exited ({self.process.exitcode})')
        return None

    def send(self, task):
        self.connection.send(task)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.is_alive():
                self.connection.send(None)
                self.process.join(timeout=1.)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2.)
            self.connection.close()
            self.process.close()


class LiveRetargeting:
    """Accept only fresh results from the current continuous tracking period.

    No backlog is sent to the process: while it is busy the UI keeps fitting
    MANO. The next dispatch uses that iteration's latest observation. ``accept``
    advances command history only for a result the caller will publish once.
    """

    def __init__(self, worker, previous, *, max_age=.5):
        self.worker, self.previous = worker, previous.copy()
        self.max_age = max_age
        self.generation = 0
        self.tracking = False
        self.observed_at = self.accepted_at = None
        self.pending = None
        self.accepted = self.discarded = 0

    def observe(self, tracking, captured_at):
        discontinuity = (self.observed_at is not None
                         and captured_at - self.observed_at > self.max_age)
        if tracking != self.tracking or discontinuity:
            self.generation += 1
        self.tracking, self.observed_at = tracking, captured_at

    def accept(self, now):
        reply = self.worker.poll()
        if reply is None:
            return None
        task, self.pending = self.pending, None
        if task is None or (reply.sequence, reply.generation, reply.captured_at) != (
                task.sequence, task.generation, task.captured_at):
            raise RuntimeError('Unexpected Botyard solver reply')
        if (not self.tracking or reply.generation != self.generation
                or not 0 <= now - reply.captured_at <= self.max_age):
            self.discarded += 1
            return None
        if (reply.actuators.shape != self.previous.shape
                or not np.isfinite(reply.q).all() or not np.isfinite(reply.actuators).all()):
            raise ValueError('Invalid retargeted joint position')
        self.previous = reply.actuators.copy()
        self.accepted_at = reply.captured_at
        self.accepted += 1
        return reply

    def submit(self, sequence, captured_at, joints, vertices, now):
        if (not self.worker.ready or self.pending is not None or not self.tracking
                or not 0 <= now - captured_at <= self.max_age):
            return False
        elapsed = captured_at - self.accepted_at if self.accepted_at is not None else 1 / 30
        if elapsed <= 0:
            return False
        task = PoseTask(sequence, self.generation, captured_at, joints.copy(), vertices.copy(),
                        self.previous.copy(), min(elapsed, .1))
        self.worker.send(task)
        self.pending = task
        return True
