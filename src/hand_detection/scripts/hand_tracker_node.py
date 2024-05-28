#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import mediapipe as mp
import cv2
from cv_bridge import CvBridge
from hand_detection.mpDraw import draw_landmarks_on_image as MPDraw
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

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

        # Create a HandLandmarker object.
        package_path = os.path.dirname(__file__)
        model_path = os.path.join(package_path, 'resources', 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)


    def listener_callback(self, msg):
        # load input frame and convert to mediapipe image formate
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect hand landmarks from the input image.
        detection_result = self.detector.detect(mp_image)

        # draw landmarks over original image
        annotated_image = MPDraw(mp_image.numpy_view(), detection_result)

        # convert from mp format to cv2 format to ros formate
        cv2_formatted_img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        processed_frame_msg = self.bridge.cv2_to_imgmsg(cv2_formatted_img, 'bgr8')
        self.publisher.publish(processed_frame_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
