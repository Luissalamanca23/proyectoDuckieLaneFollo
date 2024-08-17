#!/usr/bin/env python3
import cv2
import os
import rospy
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from duckietown.dtros import DTROS, NodeType
import numpy as np

class LaneFollowNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.node_name = node_name

        # Initialize parameters
        self.pub_img = None
        self.run = True

        # HSV color values for masking
        self.lower_bound = np.array([20, 45, 25])
        self.upper_bound = np.array([35, 255, 255]) 

        # Driving parameters
        self.drive = True
        self.speed = 0.2
        self.omega = 0
        self.size_ratio = 0.8   # Distance from the center of the Duckiebot to the dotted line

        # PID control parameters
        self.prev_time = rospy.Time.now().to_sec()
        self.prev_error = 0.0
        self.integral = 0.0

        # Initial PID values (Kp, Ki, Kd)
        self.Kp = 1.0
        self.Ki = 0.0
        self.Kd = 0.0
        self.prev_diff = None

        # Ziegler-Nichols parameters (tuning)
        self.ultimate_gain = 1.0  # Ku
        self.ultimate_period = 2.0  # Pu

        # Subscribers
        img_topic = f"""/{os.environ['VEHICLE_NAME']}/camera_node/image/compressed"""
        self.img_sub = rospy.Subscriber(img_topic, CompressedImage, self.cb_img, queue_size=1)

        # Publishers
        self.img_publisher = rospy.Publisher('/masked_image/compressed', CompressedImage, queue_size=1)
        twist_topic = f"/{os.environ['VEHICLE_NAME']}/car_cmd_switch_node/cmd"
        self.twist_publisher = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

    def cb_img(self, msg):
        """Callback for processing the incoming image from the Duckiebot camera."""
        # Convert the image from the camera and apply HSV mask
        data_arr = np.frombuffer(msg.data, np.uint8)
        col_img = cv2.imdecode(data_arr, cv2.IMREAD_COLOR)
        crop_img = col_img[len(col_img) // 3:]  # Crop the image for processing efficiency
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        imagemask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)

        # Find all yellow lane markings
        contours, _ = cv2.findContours(imagemask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest yellow stripe
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)

            # Ignore the largest stripe if it is too close to the bot
            if y > 200:
                contours.remove(largest)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest)

            # Draw visualization on the image
            image = cv2.drawContours(crop_img, [largest], -1, (0, 255, 0), 3)
            image = cv2.line(image, (x, y + h // 2), (x + int((self.size_ratio * (y + h))), y + h), (0, 255, 0), 2)

            imx, imy, _ = image.shape
            r = imy - y
            image = cv2.circle(image, (int(imx // 2 - r * np.sin(self.omega)), int(imy - r * np.cos(self.omega))), 4, (0, 0, 255), -1)

            # Draw center line for reference
            for i in range(len(image)):
                image[i][len(image[i]) // 2] = [255, 0, 0]

            self.pub_img = image

            # Move the bot if drive is enabled
            if self.drive:
                self.pid(x, y + h // 2, imx // 2)

    def img_pub(self):
        """Publish the processed image."""
        if self.pub_img is not None:
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', self.pub_img)[1]).tobytes()
            self.img_publisher.publish(msg)

    def twist_pub(self):
        """Publish the twist command to control the Duckiebot's movement."""
        if self.drive:
            msg = Twist2DStamped()
            msg.v = self.speed
            msg.omega = self.omega
            self.twist_publisher.publish(msg)

    def pid(self, x, y, goal):
        """PID control to calculate the steering angle (omega)."""
        scale_for_pixel_area = 0.02
        error = ((x + int((self.size_ratio * y))) - goal) * scale_for_pixel_area
        
        # Calculate the time difference
        current_time = rospy.Time.now().to_sec()
        dt = current_time - self.prev_time if self.prev_time else 1e-3  # Avoid division by zero
        self.prev_time = current_time

        # Update integral and derivative
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        # Ziegler-Nichols tuning (adjusting Kp, Ki, Kd)
        self.Kp = 0.6 * self.ultimate_gain
        self.Ki = 1.2 * self.ultimate_gain / self.ultimate_period
        self.Kd = 3 * self.ultimate_gain * self.ultimate_period / 40

        # Calculate the new omega value
        self.omega = -(self.Kp * error + self.Ki * self.integral + self.Kd * derivative)
        rospy.loginfo(f"PID error: {error}, Omega: {self.omega}, Kp: {self.Kp}, Ki: {self.Ki}, Kd: {self.Kd}")

    def shutdown(self):
        """Custom shutdown procedure to stop the Duckiebot."""
        rospy.loginfo("Shutting down lane follow node...")
        self.drive = False
        self.twist_pub()  # Ensure the robot stops
        rospy.loginfo("Lane follow node stopped.")


if __name__ == '__main__':
    # Initialize the node
    node = LaneFollowNode(node_name='lane_follow_node')

    # Setup a shutdown hook to stop the robot safely
    rospy.on_shutdown(node.shutdown)

    rate = rospy.Rate(10)  # 10Hz
    while not rospy.is_shutdown() and node.run:
        node.img_pub()
        node.twist_pub()
        rate.sleep()