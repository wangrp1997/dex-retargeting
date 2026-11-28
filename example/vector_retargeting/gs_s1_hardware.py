"""Relay tracked Botyard targets to the existing default dexhand_controller."""
import argparse
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
from std_msgs.msg import Float64MultiArray

from gs_s1_hardware_commands import FINGER_JOINTS, PositionRelay, WRIST_JOINTS


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
        self.active, self.status_time = False, 0.
        namespace = '/' + args.controller_namespace.strip('/') if args.controller_namespace.strip('/') else ''
        controller = namespace + '/dexhand_controller'
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
        if self.parameters_future is not None and self.parameters_future.done():
            response = self.parameters_future.result()
            self.parameters_future = None
            if response and response.values:
                joints = list(response.values[0].string_array_value)
                try:
                    self.relay = PositionRelay(joints, self.limits)
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
        if self.relay is None and self.parameters_future is None and self.parameters_client.service_is_ready():
            request = GetParameters.Request()
            request.names = ['joints']
            self.parameters_future = self.parameters_client.call_async(request)
        if self.status_future is None and self.status_client.service_is_ready():
            self.status_future = self.status_client.call_async(ListControllers.Request())

    def feedback(self, message):
        if self.relay is not None:
            try:
                self.relay.set_feedback(message.name, message.position, monotonic())
            except (KeyError, ValueError) as error:
                self.get_logger().warning(f'Invalid joint feedback: {error}', throttle_duration_sec=2.)

    def target(self, message):
        if self.relay is not None:
            try:
                age = (self.get_clock().now().nanoseconds
                       - message.header.stamp.sec * 10**9 - message.header.stamp.nanosec) / 1e9
                if not 0 <= age <= self.relay.target_timeout:
                    self.relay.target = None
                    return
                self.relay.set_target(message.name, message.position, monotonic())
            except (KeyError, ValueError) as error:
                self.get_logger().warning(f'Invalid tracked target: {error}', throttle_duration_sec=2.)

    def publish(self):
        if self.relay is None:
            return
        if not self.active or monotonic() - self.status_time > 3.:
            self.relay.reset()
            return
        command = self.relay.step(monotonic())
        if command is not None:
            self.publisher.publish(Float64MultiArray(data=command))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hand', choices=('right', 'left'), default='left')
    parser.add_argument('--controller-namespace', default='')
    args = parser.parse_args()
    rclpy.init(args=[])
    node = HardwareRelay(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
