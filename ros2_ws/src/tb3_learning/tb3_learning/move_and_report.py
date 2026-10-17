import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

FORWARD_SPEED = 0.1       # 前进线速度，单位 m/s
FORWARD_DURATION = 3.0    # 前进时长，秒
TIMER_PERIOD = 0.1        # 定时器周期，0.1 秒 = 10 Hz


class MoveAndReport(Node):

    def __init__(self):
        super().__init__('move_and_report')

        # 发布速度命令（和避障节点同款：/cmd_vel 上发 TwistStamped）
        self.velocity_publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # 订阅里程计，收到数据就调用 odom_callback
        self.odom_subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # 定时器：每 0.1 秒调用一次 timer_callback
        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)

        self.start_time = None
        self.stopped = False

    def timer_callback(self):
        if self.start_time is None:
            return  # 里程计还没连上，先不发命令
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if elapsed < FORWARD_DURATION:
            # 阶段一：发前进命令
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.twist.linear.x = FORWARD_SPEED
            self.velocity_publisher.publish(cmd)
        elif not self.stopped:
            # 阶段二：只发一次全零 Twist，就是"主动停止"
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            self.velocity_publisher.publish(cmd)
            self.stopped = True
            self.get_logger().info('已发送停止命令')

    def odom_callback(self, msg: Odometry):
        if self.start_time is None:
            self.start_time = self.get_clock().now()
            self.get_logger().info('里程计已连接，开始计时前进')
        pos = msg.pose.pose.position
        self.get_logger().info(f'当前位置: x={pos.x:.3f} m, y={pos.y:.3f} m')


def main(args=None):
    rclpy.init(args=args)
    node = MoveAndReport()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C 时 SIGINT 可能已先关闭 ROS 上下文，补发零速只能尽力而为
        try:
            node.velocity_publisher.publish(TwistStamped())
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main() 