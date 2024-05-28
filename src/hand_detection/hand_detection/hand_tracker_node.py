#!/usr/bin/env python3
# Node to track hand landmarks and determine if an open palm is present given a ros image
# If no hand detected at all or if the hand is not an open palm, publishes false to openPalm_detection topic
# If there is a hand in frame AND it is an open palm, publishes true
#

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool 
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
        self.img_publisher = self.create_publisher(Image, '/processed_frames', 10)
        self.bool_publisher = self.create_publisher(Bool, '/openPalm_detection', 10)
        self.bridge = CvBridge()

        # Create a HandLandmarker object.
        package_path = os.path.dirname(__file__)
        model_path = os.path.join(package_path, 'resources', 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)


    def listener_callback(self, msg):
        # load input frame and convert to mediapipe image format
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect hand landmarks from the input image.
        detection_result = self.detector.detect(mp_image)

        # publish annotated image: draw landmarks, convert from mp -> cv2 -> ros_img format, publish
        annotated_image = MPDraw(mp_image.numpy_view(), detection_result)
        cv2_formatted_img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        processed_frame_msg = self.bridge.cv2_to_imgmsg(cv2_formatted_img, 'bgr8')
        self.img_publisher.publish(processed_frame_msg)

        # insert keypoints into numpy array for processing: assumes only one hand detected
        if len(detection_result.hand_landmarks) >= 1: # at least one hand detected 
            keypoints = np.zeros((3,len(detection_result.hand_landmarks[0])))
            for i, keypoint in enumerate(detection_result.hand_landmarks[0]):
                keypoints[:,i] = np.array([keypoint.x, keypoint.y, keypoint.z])

            # see if fingertips extend some distance beyond palm
            palm_open = True
            palm_indices = [5, 9, 13, 17] # indices of bases of fingers
            finger_indices = [8, 12, 16, 20] # indices of fingertip keypoints
            finger_extended_cutoff = 0.5
            for i, fidx in enumerate(finger_indices): # fidx = fingertip index
                pidx = palm_indices[i] # pidx = palm/finger base index
                finger_keypoint = keypoints[:,fidx]
                palm_keypoint = keypoints[:,pidx]
                finger_vector = (palm_keypoint - finger_keypoint)
                wrist2palm_vector = (keypoints[:,0] - palm_keypoint)
                
                # see what results this dot product gives to determine finger cutoff:
                val = np.inner(finger_vector, wrist2palm_vector)/np.inner(wrist2palm_vector, wrist2palm_vector)
                #print('finger ', i, ': ', val)
                if val < finger_extended_cutoff:
                    palm_open = False
        else:
            palm_open = False

        self.bool_publisher.publish(Bool(data=palm_open))

def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
