#!/usr/bin/env python3
import dt_apriltags
import cv2
import tf
import os
import rospy
from duckietown.dtros import DTROS, NodeType
import numpy as np

from sensor_msgs.msg import CameraInfo
from duckietown_msgs.msg import LEDPattern
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Float32, Int32, ColorRGBA, Bool

class AprilTagNode(DTROS):

    def __init__(self, node_name):
        super(AprilTagNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.node_name = node_name
        self.camera_calibration = None
        self.camera_parameters = None
        self.safe_to_run_program = False
        self._tf_bcaster = tf.TransformBroadcaster()

        # Initialize image variables
        self.grey_img = np.array([])
        self.col_img = None
        self.curr_msg = None
        self.detector = dt_apriltags.Detector()

        # State variables
        self.run = True
        self.curr_col = "WHITE"
        self.prev_tag = 0
        self.dist_from_april = float('inf')
        self.error_from_april = 0
        self.april_priority = -1

        # Map of sign ID to color
        self.sign_col_map = {
            153: "BLUE", 58: "BLUE", 11: "BLUE", 62: "BLUE",   # T-intersection signs
            24: "RED", 26: "RED",                              # Stop signs
            57: "GREEN", 200: "GREEN", 94: "GREEN", 93: "GREEN" # UofA Tags
        }

        # Subscribers
        img_topic = f"""/{os.environ['VEHICLE_NAME']}/camera_node/image/compressed"""
        info_topic = f"""/{os.environ['VEHICLE_NAME']}/camera_node/camera_info"""
        self.img_sub = rospy.Subscriber(img_topic, CompressedImage, self.cb_img, queue_size=1)
        self.subscriber_camera_info = rospy.Subscriber(info_topic, CameraInfo, self.camera_info_callback, queue_size=1)
        self.kill_sub = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/shutdown", Bool, self.cb_kill, queue_size=1)
        self.april_priority_sub = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/april_priority", Int32, self.cb_april_priority, queue_size=1)

        # Publishers
        self.pub = rospy.Publisher("/" + os.environ['VEHICLE_NAME'] + '/grey_img/compressed', CompressedImage, queue_size=1)
        self.pub_led = rospy.Publisher("/" + os.environ['VEHICLE_NAME'] + "/led_emitter_node/led_pattern", LEDPattern, queue_size=1)
        self.dist_from_pub = rospy.Publisher("/" + os.environ['VEHICLE_NAME'] + '/dist_from_april', Float32, queue_size=1)
        self.april_id = rospy.Publisher("/" + os.environ['VEHICLE_NAME'] + '/april_id', Int32, queue_size=1)
        self.april_x_error = rospy.Publisher("/" + os.environ['VEHICLE_NAME'] + '/april_x_error', Int32, queue_size=1)

        # Initialize LEDs to white
        self.publishLEDs(1.0, 1.0, 1.0)

    def cb_april_priority(self, msg):
        self.april_priority = msg.data

    def cb_kill(self, msg):
        self.run = msg.data

    def camera_info_callback(self, msg):
        self.camera_calibration = msg
        print("== Calibrating Camera ==")

        curr_raw_image_height = 640
        curr_raw_image_width = 480

        scale_matrix = np.ones(9)
        if self.camera_calibration.height != curr_raw_image_height or self.camera_calibration.width != curr_raw_image_width:
            scale_width = float(curr_raw_image_width) / self.camera_calibration.width
            scale_height = float(curr_raw_image_height) / self.camera_calibration.height
            scale_matrix[0] *= scale_width
            scale_matrix[2] *= scale_width
            scale_matrix[4] *= scale_height
            scale_matrix[5] *= scale_height

        self.tag_size = 0.065
        rect_K, _ = cv2.getOptimalNewCameraMatrix(
            (np.array(self.camera_calibration.K)*scale_matrix).reshape((3, 3)),
            self.camera_calibration.D,
            (640, 480),
            1.0
        )
        self.camera_parameters = (rect_K[0, 0], rect_K[1, 1], rect_K[0, 2], rect_K[1, 2])

        try:
            self.subscriber_camera_info.shutdown()
            self.safe_to_run_program = True
            print("== Camera Info Subscriber successfully killed ==")
        except BaseException:
            pass

    def cb_img(self, msg):
        data_arr = np.frombuffer(msg.data, np.uint8)
        col_img = cv2.imdecode(data_arr, cv2.IMREAD_COLOR)
        grey_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
        self.grey_img = grey_img[1 * len(col_img) // 4: 2 * len(col_img) // 3]
        self.col_img = col_img[1 * len(col_img) // 4: 2 * len(col_img) // 3]
        self.curr_msg = msg

    def img_pub(self):
        if self.grey_img.any():
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', self.grey_img)[1]).tobytes()
            self.pub.publish(msg)

    def publishLEDs(self, red, green, blue):
        set_led_cmd = LEDPattern()
        for i in range(5):
            rgba = ColorRGBA()
            rgba.r = red
            rgba.g = green
            rgba.b = blue
            rgba.a = 1.0
            set_led_cmd.rgb_vals.append(rgba)
        self.pub_led.publish(set_led_cmd)

    def pub_april_x_error(self):
        msg = Int32()
        msg.data = self.error_from_april
        self.april_x_error.publish(msg)

    def dist_pub(self):
        msg = Float32()
        msg.data = self.dist_from_april * 2  # Adjusted distance
        self.dist_from_pub.publish(msg)

    def _matrix_to_quaternion(self, r):
        T = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)), dtype=np.float64)
        T[0:3, 0:3] = r
        return tf.transformations.quaternion_from_matrix(T)

    def detect_tag(self):
        if not self.safe_to_run_program:
            return

        img = self.col_img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray, True, self.camera_parameters, self.tag_size)

        print("[INFO] {} total AprilTags detected".format(len(tags)))

        if len(tags) == 0:
            self.change_led_to("WHITE")
            self.curr_col = "WHITE"
            self.img_pub()
            return

        closest_col = "WHITE"
        closest = 0

        for tag in tags:
            (ptA, ptB, ptC, ptD) = tag.corners
            diff = abs(ptA[0] - ptB[0])
            ptA, ptB, ptC, ptD = map(lambda pt: (int(pt[0]), int(pt[1])), (ptA, ptB, ptC, ptD))

            line_col = (125, 125, 0)
            cv2.line(img, ptA, ptB, line_col, 2)
            cv2.line(img, ptB, ptC, line_col, 2)
            cv2.line(img, ptC, ptD, line_col, 2)
            cv2.line(img, ptD, ptA, line_col, 2)

            (cX, cY) = (int(tag.center[0]), int(tag.center[1]))

            txt_col = (25, 25, 200)
            tag_id = tag.tag_id
            cv2.putText(img, str(tag_id), (cX - 9, cY + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_col, 2)

            if diff > closest:
                closest = diff
                closest_col = self.sign_to_col(tag_id)

            self.q = self._matrix_to_quaternion(tag.pose_R)
            self.p = tag.pose_t.T[0]

            self._tf_bcaster.sendTransform(
                self.p.tolist(),
                self.q.tolist(),
                self.curr_msg.header.stamp,
                "tag/{:s}".format(str(tag.tag_id)),
                self.curr_msg.header.frame_id,
            )

        self.change_led_to(closest_col)
        self.curr_col = closest_col
        self.img_pub()

    def change_led_to(self, new_col):
        colors = {"RED": (1.0, 0.0, 0.0), "GREEN": (0.0, 1.0, 0.0), "BLUE": (0.0, 0.0, 1.0), "WHITE": (1.0, 1.0, 1.0)}
        self.publishLEDs(*colors.get(new_col, (1.0, 1.0, 1.0)))

    def sign_to_col(self, tag_id):
        return self.sign_col_map.get(tag_id, "WHITE")

if __name__ == '__main__':
    node = AprilTagNode(node_name='april_tag_detector')
    rate = rospy.Rate(1)
    while not rospy.is_shutdown() and node.run:
        node.change_led_to(node.curr_col)
        node.detect_tag()
        node.pub_april_x_error()
        node.dist_pub()
        rate.sleep()
