"""Relay tracked Botyard targets to the existing default dexhand_controller."""
import argparse
import json
import os
from pathlib import Path
from time import monotonic
import xml.etree.ElementTree as ET

import rclpy
import xacro
from ament_index_python.packages import get_package_share_directory
from controller_manager_msgs.srv import ListControllers
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from gs_s1_hardware_commands import FINGER_JOINTS, PositionRelay, WRIST_JOINTS
from gs_s1_botyard_version import botyard_version


def joint_limits(hand):
    path = Path(get_package_share_directory('dexhand_description')) / 'urdf/dexhand.urdf.xacro'
    document = xacro.process_file(str(path), mappings=dict(
        use_left_hand=str(hand == 'left').lower(), sub_version='2',
        use_ros2_control='false', use_gazebo='false'))
    root = ET.fromstring(document.toxml())
    limits = {}
    for name in WRIST_JOINTS + FINGER_JOINTS:
        element = root.find(f"joint[@name='{name}']/limit")
        limits[name] = tuple(float(element.get(key)) for key in ('lower', 'upper', 'velocity'))
    return limits


class HardwareRelay(Node):
    def __init__(self, args):
        super().__init__('botyard_hardware_relay', namespace='/gs_s1_preview')
        self.args, self.limits = args, joint_limits(args.hand)
        self.relay = None
        self.session_log = None
        self.target_sequence = 0
        self.target_stamp_ns = None
        self.publish_state = None
        self.active, self.status_time = False, 0.
        namespace = '/' + args.controller_namespace.strip('/') if args.controller_namespace.strip('/') else ''
        controller = namespace + '/dexhand_controller'
        log_dir = getattr(args, 'log_dir', None)
        if log_dir is not None:
            from gs_s1_hardware_log import HardwareSessionLog
            self.session_log = HardwareSessionLog(log_dir, dict(
                hand=args.hand, controller=controller, ros_domain_id=os.environ.get('ROS_DOMAIN_ID', '0'),
                botyard_opposition=getattr(args, 'botyard_opposition', 'v4_2'),
                botyard_shape=getattr(args, 'botyard_shape', 'v4_2'),
                algorithm_version=(botyard_version(getattr(args, 'botyard_opposition', 'v4_2'))
                                   if getattr(args, 'botyard_shape', 'v4_2') == 'v4_2' else None),
                command_protocol='18 positions; controller fills default velocity/effort/tactile',
                expected_controller_velocity_rad_s=1.5708, relay_max_velocity_rad_s=1.5708,
                relay_timer_period_s=.02, relay_recovery_policy='preserve_last_output_on_target_timeout',
                io_trace_requested=getattr(args, 'log_session_dir', None) is not None,
                note='No images or video. Publication is not physical application.'),
                directory=getattr(args, 'log_session_dir', None))
            self.get_logger().info(f'Hardware diagnostics: {self.session_log.directory}')
            self.create_subscription(String, '/gs_s1_preview/pipeline_timing', self.pipeline_timing, 20)
            try:
                from dexhand_interfaces.msg import HandSample
            except ImportError:
                self.record('hardware_sample_unavailable', reason='HandSample message is not installed')
                self.get_logger().warning('HandSample unavailable; logs will contain joint_states feedback only')
            else:
                self.create_subscription(HandSample, namespace + '/dexhand_hardware/state_sample',
                                         self.hardware_sample, qos_profile_sensor_data)
        self.publisher = self.create_publisher(Float64MultiArray, controller + '/commands', 1)
        self.create_subscription(JointState, namespace + '/joint_states', self.feedback, qos_profile_sensor_data)
        self.create_subscription(JointState, '/gs_s1_preview/tracked_joint_targets', self.target, 1)
        self.parameters_client = self.create_client(GetParameters, controller + '/get_parameters')
        self.status_client = self.create_client(ListControllers, namespace + '/controller_manager/list_controllers')
        self.parameters_future = self.status_future = None
        self.create_timer(1., self.poll_controller)
        self.create_timer(.02, self.publish)
        self.get_logger().info(f'Botyard {args.hand} hand -> {controller}/commands; '
                               'waiting for the default controller, joint feedback and tracked targets')

    def poll_controller(self):
        if self.session_log is not None and self.session_log.error is not None:
            self.get_logger().error(f'Hardware diagnostic writer: {self.session_log.error}', throttle_duration_sec=5.)
        if self.parameters_future is not None and self.parameters_future.done():
            response = self.parameters_future.result()
            self.parameters_future = None
            if response and response.values:
                joints = list(response.values[0].string_array_value)
                try:
                    self.relay = PositionRelay(joints, self.limits)
                    self.record('controller_configuration', joint_names=joints,
                                joint_limits=self.limits, max_velocity_rad_s=self.relay.max_velocity)
                    self.get_logger().info('Controller joint order verified; wrist pose follows initial feedback')
                except ValueError as error:
                    self.get_logger().error(str(error))
        if self.status_future is not None and self.status_future.done():
            response = self.status_future.result()
            self.status_future = None
            self.active = bool(response and any(
                c.name == 'dexhand_controller' and c.state == 'active'
                and c.type == 'dexhand_controller/DexhandController' for c in response.controller))
            self.status_time = monotonic()
            self.record('controller_status', active=self.active)
        if self.relay is None and self.parameters_future is None and self.parameters_client.service_is_ready():
            request = GetParameters.Request()
            request.names = ['joints']
            self.parameters_future = self.parameters_client.call_async(request)
        if self.status_future is None and self.status_client.service_is_ready():
            self.status_future = self.status_client.call_async(ListControllers.Request())

    def feedback(self, message):
        self.record('feedback', source_stamp_ns=message.header.stamp.sec * 10**9 + message.header.stamp.nanosec,
                    joint_names=list(message.name), positions=list(message.position),
                    velocities=list(message.velocity), efforts=list(message.effort))
        if self.relay is not None:
            try:
                self.relay.set_feedback(message.name, message.position, monotonic())
            except (KeyError, ValueError) as error:
                self.get_logger().warning(f'Invalid joint feedback: {error}', throttle_duration_sec=2.)

    def target(self, message):
        self.target_sequence += 1
        source_stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
        received_ns = self.get_clock().now().nanoseconds
        age = (received_ns - source_stamp) / 1e9
        fields = dict(target_sequence=self.target_sequence, source_stamp_ns=source_stamp, source_age_s=age,
                      joint_names=list(message.name), positions=list(message.position))
        if self.relay is not None:
            try:
                if not 0 <= age <= self.relay.target_timeout:
                    self.relay.target = None
                    self.record('target', accepted=False, reason='expired_or_future_stamp', **fields)
                    return
                self.relay.set_target(message.name, message.position, monotonic() - age)
                self.target_stamp_ns = source_stamp
                self.record('target', accepted=True, **fields)
            except (KeyError, ValueError) as error:
                self.record('target', accepted=False, reason=str(error), **fields)
                self.get_logger().warning(f'Invalid tracked target: {error}', throttle_duration_sec=2.)
        else:
            self.record('target', accepted=False, reason='controller_not_ready', **fields)

    def publish(self):
        if self.relay is None:
            self.set_publish_state('waiting_for_controller')
            return
        if not self.active or monotonic() - self.status_time > 3.:
            self.relay.reset()
            self.set_publish_state('controller_inactive_or_status_expired')
            return
        now = monotonic()
        command = self.relay.step(now)
        if command is not None:
            self.publisher.publish(Float64MultiArray(data=command))
            self.set_publish_state('forwarding')
            if self.session_log is not None:
                errors = [self.relay.target[n] - self.relay.command[n] for n in FINGER_JOINTS]
                self.record('command', joint_names=list(self.relay.joints), positions=command,
                            target_sequence=self.target_sequence, source_stamp_ns=self.target_stamp_ns,
                            target_age_s=now - self.relay.target_time,
                            feedback_receipt_age_s=now - self.relay.feedback_time,
                            target_error_joint_names=list(FINGER_JOINTS), target_error_rad=errors,
                            ramp_pending_joints=[n for n, e in zip(FINGER_JOINTS, errors) if abs(e) > 1e-6])
        else:
            if self.relay.feedback is None:
                reason = 'waiting_for_feedback'
            elif self.relay.target is None:
                reason = 'waiting_for_target'
            elif not 0 <= now - self.relay.target_time <= self.relay.target_timeout:
                reason = 'target_expired'
            else:
                reason = 'feedback_expired'
            self.set_publish_state(reason)

    def record(self, event, **fields):
        if self.session_log is not None:
            self.session_log.record(event, self.get_clock().now().nanoseconds, **fields)

    def set_publish_state(self, state):
        if state != self.publish_state:
            self.record('relay_state', state=state)
            self.publish_state = state

    def hardware_sample(self, message):
        self.record('hardware_sample', source_stamp_ns=message.header.stamp.sec * 10**9 + message.header.stamp.nanosec,
                    sequence=message.sequence, valid=message.valid, configured_io_rate_hz=message.configured_io_rate_hz,
                    poll_interval_s=message.poll_interval_s, read_duration_s=message.read_duration_s,
                    read_failures=message.read_failures, joint_names=list(message.joint_names),
                    positions=list(message.positions), velocities=list(message.velocities), efforts=list(message.efforts),
                    tactile_names=list(message.tactile_names), tactile_forces=list(message.tactile_forces))

    def pipeline_timing(self, message):
        try:
            timing = json.loads(message.data)
            if not isinstance(timing, dict):
                raise ValueError('Expected timing object')
            self.record('pipeline', timing=timing)
        except ValueError as error:
            self.record('invalid_pipeline_timing', reason=str(error))

    def close_log(self):
        if self.session_log is not None:
            try:
                self.session_log.close()
                self.get_logger().info(f'Hardware diagnostics saved: {self.session_log.directory}')
            except Exception as error:
                self.get_logger().error(f'Could not finish hardware diagnostic log: {error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hand', choices=('right', 'left'), default='left')
    parser.add_argument('--controller-namespace', default='')
    parser.add_argument('--botyard-opposition', choices=('v4_2', 'inward'), default='v4_2',
                        help='Tracker mode recorded in diagnostic metadata; controller settings are unchanged')
    parser.add_argument('--botyard-shape', default='v4_2', help='Tracker shape recorded in diagnostic metadata')
    parser.add_argument('--log-dir', type=Path, help='Enable hardware diagnostics under this directory')
    parser.add_argument('--log-session-dir', type=Path, help='Session reserved by the IO tracing launch')
    args = parser.parse_args()
    if args.log_session_dir and not args.log_dir:
        parser.error('--log-session-dir requires --log-dir')
    rclpy.init(args=[])
    node = HardwareRelay(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_log()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
