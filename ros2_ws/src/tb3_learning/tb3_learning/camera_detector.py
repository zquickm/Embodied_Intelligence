import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

import numpy as np
import cv2

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
MARKER_LENGTH = 0.24   # 黑色方块物理边长（米）
AXIS_LEN = 0.2         # 坐标轴长度（米），比默认 0.1 更好看
CORNER_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0)]
CORNER_NAMES = ['A', 'B', 'C', 'D']


class CameraDetector(Node):

    def __init__(self):
        super().__init__('camera_detector')

        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)

        self.parameters = cv2.aruco.DetectorParameters_create()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

        self.frame_count = 0
        self.report_time = None
        self.last_fps = 0.0
        self.last_ids = None
        self.last_tvec = None
        self.last_corners = None

    def info_callback(self, msg: CameraInfo):
        if self.camera_matrix is not None:
            return
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.get_logger().info(
            f'内参已就绪: fx={self.camera_matrix[0,0]:.4f} '
            f'cx={self.camera_matrix[0,2]:.0f} cy={self.camera_matrix[1,2]:.0f}')

    def annotate(self, img, corners, ids, rvecs, tvecs):
        """把检测结果画到画面上：四角 + 加粗绿框 + 坐标轴 + 左上数据面板"""
        if ids is not None:
            for idx, corner in enumerate(corners):
                pts = corner.reshape(-1, 2)
                cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)],
                              True, (0, 255, 0), 3, cv2.LINE_AA)

                for i in range(4):
                    c = (int(pts[i][0]), int(pts[i][1]))
                    cv2.circle(img, c, 8, CORNER_COLORS[i], -1, cv2.LINE_AA)
                    cv2.circle(img, c, 8, (255, 255, 255), 2, cv2.LINE_AA)

                    off_x = 14 if i in (0, 3) else -30
                    off_y = -14 if i in (0, 1) else 24
                    pos = (c[0] + off_x, c[1] + off_y)
                    cv2.putText(img, CORNER_NAMES[i], pos, cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, CORNER_NAMES[i], pos, cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, CORNER_COLORS[i], 2, cv2.LINE_AA)

                if rvecs is not None and idx < len(rvecs):
                    cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs,
                                      rvecs[idx], tvecs[idx], AXIS_LEN, 3)

        # 左上角数据面板
        lines = []
        if ids is not None:
            lines.append(f'id = {ids.flatten().tolist()}')
            if tvecs is not None and len(tvecs):
                t = tvecs[0].ravel()
                lines.append(f'x = {t[0]:+.3f} m')
                lines.append(f'y = {t[1]:+.3f} m')
                lines.append(f'z = {t[2]:+.3f} m')
            pts = corners[0].reshape(-1, 2)
            lines.append(f'corner A ({int(pts[0][0])},{int(pts[0][1])})')
            lines.append(f'corner C ({int(pts[2][0])},{int(pts[2][1])})')
        else:
            lines.append('NO MARKER')
        lines.append(f'FPS {self.last_fps:.1f}')

        panel_w = 320
        panel_h = 24 * len(lines) + 14
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        for i, text in enumerate(lines):
            color = (0, 255, 0) if ids is not None else (0, 0, 255)
            cv2.putText(img, text, (10, 26 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        return img

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

        rvecs = tvecs = None
        if ids is not None and self.camera_matrix is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, self.camera_matrix, self.dist_coeffs)

        img = self.annotate(img, corners, ids, rvecs, tvecs)

        cv2.imshow('camera_detector', img)
        cv2.waitKey(1)

        self.frame_count += 1
        now = self.get_clock().now()
        if self.report_time is None:
            self.report_time = now
        elif (now - self.report_time).nanoseconds >= 5e9:
            self.last_fps = self.frame_count / ((now - self.report_time).nanoseconds / 1e9)
            seen = ids.flatten().tolist() if ids is not None else '无'
            self.get_logger().info(f'近 5 秒 {self.frame_count} 帧, FPS {self.last_fps:.1f}, 最新检测: {seen}')
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