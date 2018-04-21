#!/usr/bin/env python

import datetime
from cv_bridge import CvBridge
from threading import Lock

import cv2
import rospy
from pelicannon.msg import CategorizedRegionsOfInterest
from sensor_msgs.msg import Image


class DebugNode(object):
    def __init__(self, target_fps=30):
        self._cv_br = CvBridge()

        self._ros_init()

        self._roi_lock = Lock()
        self._latest_rois = []

        self._target_fps = target_fps
        self._frames_processed = 0
        self._start_time = datetime.datetime.now()

    def _ros_init(self):
        rospy.init_node('debug')

        rospy.Subscriber('regions_of_interest', CategorizedRegionsOfInterest, self._roi_callback)
        rospy.Subscriber('cv_peek', Image, self._camera_callback)

        self._publisher = rospy.Publisher('cv_debug', Image, queue_size=10)

    def _roi_callback(self, rois):

        self._roi_lock.acquire()
        self._latest_rois = rois.regions
        self._roi_lock.release()

    def _camera_callback(self, image):

        time_delta = datetime.datetime.now() - self._start_time
        seconds_run = time_delta.seconds + time_delta.microseconds / 1000000.

        fps = self._frames_processed / seconds_run

        if fps > self._target_fps:
            return

        frame = self._cv_br.imgmsg_to_cv2(image, desired_encoding="passthrough")

        self._roi_lock.acquire()
        for roi in self._latest_rois:
            if roi.category == "body":
                cv2.rectangle(frame, (roi.x, roi.y),
                              (roi.x + roi.w, roi.y + roi.h), (0, 255, 0), 2)
            elif roi.category == "motion":
                cv2.rectangle(frame, (roi.x, roi.y),
                              (roi.x + roi.w, roi.y + roi.h), (255, 0, 0), 2)
        self._roi_lock.release()

        self._publisher.publish(self._cv_br.cv2_to_imgmsg(frame, encoding="passthrough"))
        self._frames_processed += 1


if __name__ == "__main__":
    node = DebugNode()
    rospy.spin()
