"""Named Botyard targets to the default dexhand_controller position protocol."""
import math


FINGER_JOINTS = (
    'THJ4', 'THJ3', 'THJ2', 'THJ1',
    'FFJ4', 'FFJ3', 'FFJ2', 'MFJ4', 'MFJ3', 'MFJ2',
    'RFJ4', 'RFJ3', 'RFJ2', 'LFJ4', 'LFJ3', 'LFJ2',
)
WRIST_JOINTS = ('FAJ3', 'FAJ1')


def select_positions(names, positions, required):
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError('Joint names and positions must be unique and paired')
    values = dict(zip(names, positions))
    selected = {name: float(values[name]) for name in required}
    if not all(math.isfinite(value) for value in selected.values()):
        raise ValueError('Joint positions must be finite')
    return selected


class PositionRelay:
    """Preserve wrist pose and approach named finger targets from feedback.

    Only actively tracked targets enter this relay. Missing targets suspend
    publication while retaining the last output setpoint, which the active
    controller also retains. Delayed feedback must not rewind that setpoint on
    resumption. Feedback or controller loss resets the output history, so a new
    control session starts from the measured pose. The 18-value output uses the
    controller's actual joint order and default motion settings.
    """

    def __init__(self, joints, limits, *, target_timeout=.5, feedback_timeout=1., max_velocity=1.5708):
        if len(joints) != 18 or set(joints) != set(WRIST_JOINTS + FINGER_JOINTS):
            raise ValueError('Expected the default single GS S1 controller joints')
        self.joints = tuple(joints)
        self.limits = limits
        if min(target_timeout, feedback_timeout, max_velocity) <= 0:
            raise ValueError('Timeouts and maximum velocity must be positive')
        self.target_timeout, self.feedback_timeout = target_timeout, feedback_timeout
        self.max_velocity = max_velocity
        self.feedback = self.target = None
        self.feedback_time = self.target_time = None
        self.reset()

    def reset(self):
        self.command = None
        self.last_time = None

    def set_feedback(self, names, positions, now):
        self.feedback = None
        self.feedback = select_positions(names, positions, self.joints)
        self.feedback_time = now

    def set_target(self, names, positions, now):
        self.target = None
        target = select_positions(names, positions, FINGER_JOINTS)
        for name, value in target.items():
            lo, hi, _ = self.limits[name]
            if value < lo - 1e-5 or value > hi + 1e-5:
                raise ValueError(f'{name} target is outside its URDF limits')
            target[name] = min(hi, max(lo, value))
        self.target, self.target_time = target, now

    def step(self, now):
        if (self.feedback is None
                or not 0 <= now - self.feedback_time <= self.feedback_timeout):
            self.reset()
            return None
        if self.target is None or not 0 <= now - self.target_time <= self.target_timeout:
            # A target timeout does not deactivate the controller: its held
            # setpoint remains our last output, not the delayed measured pose.
            # Pause the integration clock too, so silence earns no larger step.
            self.last_time = now
            return None
        if self.command is None:
            self.command = self.feedback.copy()
            self.last_time = now
        dt = min(.1, max(0., now - self.last_time))
        for name in FINGER_JOINTS:
            step = min(self.max_velocity, self.limits[name][2]) * dt
            delta = self.target[name] - self.command[name]
            self.command[name] += min(step, max(-step, delta))
        self.last_time = now
        return [self.command[name] for name in self.joints]
