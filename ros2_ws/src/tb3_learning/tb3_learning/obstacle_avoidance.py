import math

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


SAFE_DISTANCE = 0.5
FORWARD_SPEED = 0.1
TURN_SPEED = 0.5
FRONT_HALF_ANGLE = math.radians(20.0)


class ObstacleAvoidance(Node):

    def __init__(self) -> None:
        super().__init__('obstacle_avoidance')

        # 创建速度命令发布器
        self.velocity_publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10,
        )

        # 订阅激光雷达数据
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info('自动避障节点已启动')

    def scan_callback(self, scan: LaserScan) -> None:
        """接收激光雷达数据并控制小车。"""
        front_distances = []

        for index, distance in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            normalized_angle = math.atan2(math.sin(angle), math.cos(angle))

            # 只保留小车正前方左右各20度的有效数据
            if (
                abs(normalized_angle) <= FRONT_HALF_ANGLE
                and not math.isnan(distance)
                and distance >= scan.range_min
            ):
                front_distances.append(distance)

        # 没有有效数据时按前方存在障碍处理
        front_distance = min(front_distances, default=0.0)

        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()

        if front_distance < SAFE_DISTANCE:
            # 前方存在障碍：停止前进并向左转
            command.twist.linear.x = 0.0
            command.twist.angular.z = TURN_SPEED
        else:
            # 前方安全：缓慢直行
            command.twist.linear.x = FORWARD_SPEED
            command.twist.angular.z = 0.0

        self.velocity_publisher.publish(command)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObstacleAvoidance()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 退出前发送一次停止命令
        stop_command = TwistStamped()
        stop_command.header.stamp = node.get_clock().now().to_msg()
        node.velocity_publisher.publish(stop_command)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()