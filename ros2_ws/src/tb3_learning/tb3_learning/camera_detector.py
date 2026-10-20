import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

import numpy as np
import cv2

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
MARKER_LENGTH = 0.24   # 黑色方块物理边长（米），不是板的 0.3


class CameraDetector(Node):

    def __init__(self):
        super().__init__('camera_detector')

        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        self.camera_matrix = None      # 内参从 camera_info 取，不写死
        self.dist_coeffs = None
        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)

        self.parameters = cv2.aruco.DetectorParameters_create()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

        self.frame_count = 0
        self.report_time = None
        self.first_detection = True

    def info_callback(self, msg: CameraInfo):
        if self.camera_matrix is not None:
            return
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.get_logger().info(f'内参已就绪: fx={self.camera_matrix[0, 0]:.4f} '
                               f'cx={self.camera_matrix[0, 2]:.0f} cy={self.camera_matrix[1, 2]:.0f}')

    def image_callback(self, msg: Image):
        if msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        else:
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, DICTIONARY, parameters=self.parameters)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)   # 彩色框 + id
            if self.first_detection:
                self.first_detection = False
                self.get_logger().info(f'首次检测到标记, id={ids.flatten().tolist()}')

            if self.camera_matrix is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH, self.camera_matrix, self.dist_coeffs)
                for rvec, tvec in zip(rvecs, tvecs):
                    cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs,
                                      rvec, tvec, 0.1)          # XYZ 轴，轴长 0.1 m
                    t = tvec.ravel()
                    self.get_logger().info(f'标记位置: x={t[0]:+.3f} y={t[1]:+.3f} z={t[2]:+.3f} m')

        cv2.imshow('camera_detector', img)
        cv2.waitKey(1)

        self.frame_count += 1
        now = self.get_clock().now()
        if self.report_time is None:
            self.report_time = now
        elif (now - self.report_time).nanoseconds >= 5e9:
            seen = ids.flatten().tolist() if ids is not None else '无'
            self.get_logger().info(f'近 5 秒 {self.frame_count} 帧, 最新检测: {seen}')
            self.frame_count = 0
            self.report_time = now


def main(args=None):
    rclpy.init(args=args)
    node = CameraDetector()
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