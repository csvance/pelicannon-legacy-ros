#!/usr/bin/env python

import datetime
from cv_bridge import CvBridge
from threading import Lock

import cv2
import rospy
from geometry_msgs.msg import Vector3
from pelicannon.msg import CategorizedRegionOfInterest, CategorizedRegionsOfInterest
from sensor_msgs.msg import Image

from tracker import BodyTrackerPipeline, MotionTrackerPipeline


class ObjectDetectorNode(object):
    def __init__(self, debug=False):
        self._debug = debug
        self._initialize_pipelines()
        self._cv_br = CvBridge()

        self._ahrs_lock = Lock()
        self._ahrs = None

        self._angular_velocity_lock = Lock()
        self._angular_velocity = None

        self._last_image = None

        self._ros_init()

    def _ros_init(self):
        rospy.init_node('object_detector')
        self._publisher = rospy.Publisher('regions_of_interest', CategorizedRegionsOfInterest, queue_size=10)

        rospy.Subscriber("/webcam/image_raw", Image, self._camera_callback)
        rospy.Subscriber("euler_angles", Vector3, self._ahrs_callback)
        rospy.Subscriber('angular_velocity', Vector3, self._angular_velocity_callback)

    def _initialize_pipelines(self):
        self._body_tracker = BodyTrackerPipeline()
        self._motion_detector = MotionTrackerPipeline()

        if self._debug:
            self._video_writer = cv2.VideoWriter('output.avi', cv2.cv.FOURCC('M', 'J', 'P', 'G'), 30.0, (160, 90))
        else:
            self._video_writer = None

    def _angular_velocity_callback(self, angular_velocity):
        self._angular_velocity_lock.acquire()
        self._angular_velocity = angular_velocity
        self._angular_velocity_lock.release()

    def _ahrs_callback(self, ahrs):

        self._ahrs_lock.acquire()
        self._ahrs = ahrs
        self._ahrs_lock.release()

    def _camera_callback(self, image):

        delta_t = (image.header.stamp - self._last_image.header.stamp).to_sec() if self._last_image is not None else 0.

        time_i = datetime.datetime.now()

        frame = self._cv_br.imgmsg_to_cv2(image, desired_encoding="passthrough")

        # Create grayscale versions
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Noise reduction
        frame_grayscale = cv2.GaussianBlur(frame_grayscale, (5, 5), 0)

        # Does this improve performance? In example code in OpenCV documentation.
        frame_grayscale = cv2.equalizeHist(frame_grayscale)

        # Send data down pipeline and process results
        haarcascade_regions = self._body_tracker.process_frame(frame_grayscale)
        motion_regions = self._motion_detector.process_frame(frame_grayscale, delta_t,
                                                             yaw_velocity=self._angular_velocity.z if self._angular_velocity is not None else None)

        regions = []
        for region in haarcascade_regions:
            regions.append(CategorizedRegionOfInterest(x=region.x, y=region.y, w=region.w, h=region.h, category='body'))

        for region in motion_regions:
            regions.append(
                CategorizedRegionOfInterest(x=region.x, y=region.y, w=region.w, h=region.h, category='motion'))

        self._publisher.publish(CategorizedRegionsOfInterest(regions=regions))

        time_f = datetime.datetime.now()

        if self._debug:
            rospy.loginfo(rospy.get_caller_id() + "Frame processed in %s", time_f - time_i)

        self._last_image = image


if __name__ == '__main__':
    detector = ObjectDetectorNode()
    rospy.spin()
