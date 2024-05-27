#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import mediapipe as mp
import cv2
from cv_bridge import CvBridge
from hand_detection.mpDraw import draw_landmarks_on_image as MPDraw

class HandTrackerNode(Node):
    def __init__(self):
        super().__init__('hand_tracker_node')
        self.subscription = self.create_subscription(
            Image,
            '/video_frames',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, '/processed_frames', 10)
        self.bridge = CvBridge()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_hands.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        processed_frame_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        self.publisher.publish(processed_frame_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
