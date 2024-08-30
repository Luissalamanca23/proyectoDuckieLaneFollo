#!/usr/bin/env python3
import cv2
import os
import rospy
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from duckietown.dtros import DTROS, NodeType
import numpy as np

class lane_follow_node(DTROS):

    def __init__(self, node_name):
        super(lane_follow_node, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.node_name = node_name

        self.pub_img = None
        self.run = True

        # HSV color values to mask
        self.lower_bound = np.array([20, 45, 25])
        self.upper_bound = np.array([35, 255, 255]) 
        # Drive speed and ratio of goal vs distance from bot
        self.drive = True
        self.speed = 0.2
        self.omega = 0
        self.size_ratio = 0.8   # Distance from center of duckiebot to dotted line

        self.prev_time = rospy.get_time()
        self.prev_diff = 0.0
        self.integral = 0.0
        # PID parameters: [Kp, Ki, Kd]
        self.PID = [1.0, 0.1, 0.05]  # Ajusta estos valores según sea necesario

        # Subscribers
        img_topic = f"/{os.environ['VEHICLE_NAME']}/camera_node/image/compressed"
        self.img_sub = rospy.Subscriber(img_topic, CompressedImage, self.cb_img, queue_size=1)

        # Publishers
        self.img_publisher = rospy.Publisher('/masked_image/compressed', CompressedImage, queue_size=1)

        twist_topic = f"/{os.environ['VEHICLE_NAME']}/car_cmd_switch_node/cmd"
        self.twist_publisher = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

    def cb_img(self, msg):
        # Get the image from camera and mask over the HSV range set in init
        data_arr = np.frombuffer(msg.data, np.uint8)
        col_img = cv2.imdecode(data_arr, cv2.IMREAD_COLOR)
        crop = [len(col_img) // 3, -1]
        hsv = cv2.cvtColor(col_img, cv2.COLOR_BGR2HSV)
        imagemask = np.asarray(cv2.inRange(hsv[crop[0] : crop[1]], self.lower_bound, self.upper_bound))

        # Find all the yellow dotted lines
        contours, _ = cv2.findContours(imagemask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest yellow stripe
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        conts = [largest]

        # Ignore the largest stripe if it is too close to the bot
        if y > 200:
            largest = max(contours, key=cv2.contourArea)
            conts.append(largest)
            x, y, w, h = cv2.boundingRect(largest)
        
        # Draw visualization stuff
        image = cv2.drawContours(col_img[crop[0] : crop[1]], conts, -1, (0, 255, 0), 3)
        image = cv2.line(image, (x, y + h // 2), (x + int((self.size_ratio * (y + h))), y + h), (0, 255, 0), 2)

        imx, imy, _ = image.shape
        r = imy - y
        image = cv2.circle(image, (int((len(image[0]) // 2) - r * np.sin(self.omega)), int(imy - r * np.cos(self.omega))), 4, (0, 0, 255), -1)

        for i in range(len(image)):
            image[i][len(image[i]) // 2] = [255, 0, 0]

        self.pub_img = image

        # Only move the bot if drive is true
        if self.drive:
            # Set this to y - h//2 for English driver mode
            # Set this to y + h//2 for American driver mode
            self.pid(x, y + h // 2, len(image[i]) // 2) 
        
    def img_pub(self):
        if self.pub_img is not None:
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', self.pub_img)[1]).tobytes()

            self.img_publisher.publish(msg)

    # Control the speed and angle of the bot
    def twist_pub(self):
        if self.drive:
            msg = Twist2DStamped()
            msg.v = self.speed
            msg.omega = self.omega

            self.twist_publisher.publish(msg)

    def pid(self, x, y, goal):
        # Calculate errors
        scale_for_pixel_area = 0.02
        diff = ((x + int((self.size_ratio * y))) - goal) * scale_for_pixel_area  
        
        # Time calculations
        current_time = rospy.get_time()
        delta_time = current_time - self.prev_time
        self.prev_time = current_time

        # Proportional term
        P_term = self.PID[0] * diff

        # Integral term
        self.integral += diff * delta_time
        I_term = self.PID[1] * self.integral

        # Derivative term
        D_term = self.PID[2] * (diff - self.prev_diff) / delta_time if delta_time > 0 else 0.0

        # Update omega with the PID output
        self.omega = -(P_term + I_term + D_term)

        # Store current diff for next iteration
        self.prev_diff = diff
        
        print(f"PID Output - P: {P_term}, I: {I_term}, D: {D_term}, Omega: {self.omega}")

if __name__ == '__main__':
    # Create the node
    node = lane_follow_node(node_name='custom_lane_follow')

    rate = rospy.Rate(100) # 100 Hz
    while not rospy.is_shutdown() and node.run:
        node.img_pub()
        node.twist_pub()
        rate.sleep()
