#!/usr/bin/env python

import cv2
from tracker import BodyTrackerPipeline, MotionTrackerPipeline
import datetime
import numpy as np
from cv_bridge import CvBridge

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image


class ObjectDetector(object):
    def __init__(self, debug=False):
        self._debug = debug
        self._initialize_pipelines()
        self._cv_br = CvBridge()
        self.phi = None

        self._ros_init()

    def _ros_init(self):
        rospy.init_node('object_detector', anonymous=True)
        rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self._publisher = rospy.Publisher('object_segments', String, queue_size=10)

    def _initialize_pipelines(self):
        self._body_tracker = BodyTrackerPipeline()
        self._motion_detector = MotionTrackerPipeline()

        if self._debug:
            self._video_writer = cv2.VideoWriter('output.avi', cv2.cv.FOURCC('M', 'J', 'P', 'G'), 30.0, (160, 90))
        else:
            self._video_writer = None

    def callback(self, image):

        time_i = datetime.datetime.now()

        frame = self._cv_br.imgmsg_to_cv2(image, desired_encoding="passthrough")
        frame_original = frame.copy()

        # Create grayscale versions
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Does this improve performance? In example code in OpenCV documentation.
        frame_grayscale = cv2.equalizeHist(frame_grayscale)

        # Send data down pipeline and process results
        rectangles = self._body_tracker.process_frame(frame_grayscale)
        for rect in rectangles:
            if self._debug:
                # print(rect)
                cv2.rectangle(frame_original, (rect.x, rect.y),
                              (rect.x + rect.w, rect.y + rect.h), (0, 255, 0), 2)

        rectangles = self._motion_detector.process_frame(frame_grayscale, phi=self.phi)
        for rect in rectangles:
            if self._debug:
                # print(rect)
                cv2.rectangle(frame_original, (rect.x, rect.y),
                              (rect.x + rect.w, rect.y + rect.h), (255, 0, 0), 2)

        time_f = datetime.datetime.now()

        rospy.loginfo(rospy.get_caller_id() + "Frame processed in %s", time_f-time_i)

        if self._debug:
            self._video_writer.write(frame_original)


if __name__ == '__main__':
    detector = ObjectDetector()
    rospy.spin()
