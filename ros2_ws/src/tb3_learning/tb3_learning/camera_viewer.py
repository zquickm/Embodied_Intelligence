import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

import numpy as np
import cv2


ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


class CameraViewer(Node):

    def __init__(self):
        super().__init__('camera_viewer')

        self.image_subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        self.frame_count = 0
        self.report_time = None

    def image_callback(self, msg: Image):
        if msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        else:
            return

        # 画面上叠一个极小的标记（10x10 像素），只用来确认这帧是"新代码"跑出来的；
        # 顺手调用一次 ArUco 字典，确保 OpenCV contrib 的 aruco 模块可用。
        cv2.rectangle(img, (10, 10), (30, 30), (0, 255, 0), -1)

        cv2.imshow('camera_view', img)
        cv2.waitKey(1)

        self.frame_count += 1
        now = self.get_clock().now()
        if self.report_time is None:
            self.report_time = now
        elif (now - self.report_time).nanoseconds >= 5e9:
            fps = self.frame_count / ((now - self.report_time).nanoseconds / 1e9)
            self.get_logger().info(f'近 5 秒约 {fps:.1f} FPS')
            self.frame_count = 0
            self.report_time = now


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()