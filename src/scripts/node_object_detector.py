#!/usr/bin/env python

import cv2
import math
from tracker import BodyTrackerPipeline, MotionTrackerPipeline
import datetime
from threading import Lock
from cv_bridge import CvBridge

import rospy
from sensor_msgs.msg import Image
from pelicannon.msg import CategorizedRegionOfInterest, CategorizedRegionsOfInterest, AHRS


class ObjectDetectorNode(object):
    def __init__(self, debug=False):
        self._debug = debug
        self._initialize_pipelines()
        self._cv_br = CvBridge()

        self._last_yaw_lock = Lock()
        self._last_yaw = []

        self._ros_init()

    def _ros_init(self):
        rospy.init_node('object_detector')
        self._publisher = rospy.Publisher('regions_of_interest', CategorizedRegionsOfInterest, queue_size=10)

        rospy.Subscriber("/webcam/image_raw", Image, self._camera_callback)
        rospy.Subscriber("euler_angles", AHRS, self._ahrs_callback)

    def _initialize_pipelines(self):
        self._body_tracker = BodyTrackerPipeline()
        self._motion_detector = MotionTrackerPipeline()

        if self._debug:
            self._video_writer = cv2.VideoWriter('output.avi', cv2.cv.FOURCC('M', 'J', 'P', 'G'), 30.0, (160, 90))
        else:
            self._video_writer = None

    @staticmethod
    def angle_diff(theta_i, theta_f):
        if abs(theta_f - theta_i) <= math.pi:
            return theta_f - theta_i
        else:
            if theta_f > theta_i:
                theta_f = theta_f - 2 * math.pi
            elif theta_f < theta_i:
                theta_i = theta_i - 2 * math.pi

            return theta_f - theta_i

    def _ahrs_callback(self, ninedof):

        self._last_yaw_lock.acquire()
        if len(self._last_yaw) == 0:
            self._last_yaw.append(ninedof.yaw)
            self._last_yaw_lock.release()
            return

        self._last_yaw.append(ninedof.yaw)
        self._last_yaw_lock.release()

    def _camera_callback(self, image):

        time_i = datetime.datetime.now()

        frame = self._cv_br.imgmsg_to_cv2(image, desired_encoding="passthrough")
        frame_original = frame.copy()

        # Create grayscale versions
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Does this improve performance? In example code in OpenCV documentation.
        frame_grayscale = cv2.equalizeHist(frame_grayscale)

        # Caclulate self motion
        self._last_yaw_lock.acquire()
        if len(self._last_yaw) < 2:
            delta_phi = None
        else:
            delta_phi = ObjectDetectorNode.angle_diff(self._last_yaw[0], self._last_yaw[-1])
            self._last_yaw = []
        self._last_yaw_lock.release()

        # Send data down pipeline and process results
        haarcascade_regions = self._body_tracker.process_frame(frame_grayscale)
        motion_regions = self._motion_detector.process_frame(frame_grayscale, phi=delta_phi)

        regions = []
        for rect in haarcascade_regions:
            regions.append(CategorizedRegionOfInterest(x=rect.x, y=rect.y, w=rect.w, h=rect.h, category='body'))
            if self._debug:
                cv2.rectangle(frame_original, (rect.x, rect.y),
                              (rect.x + rect.w, rect.y + rect.h), (0, 255, 0), 2)

        for rect in motion_regions:
            regions.append(CategorizedRegionOfInterest(x=rect.x, y=rect.y, w=rect.w, h=rect.h, category='motion'))
            if self._debug:
                cv2.rectangle(frame_original, (rect.x, rect.y),
                              (rect.x + rect.w, rect.y + rect.h), (255, 0, 0), 2)

        self._publisher.publish(CategorizedRegionsOfInterest(regions=regions))

        time_f = datetime.datetime.now()

        if self._debug:
            rospy.loginfo(rospy.get_caller_id() + "Frame processed in %s", time_f - time_i)
            self._video_writer.write(frame_original)


if __name__ == '__main__':
    detector = ObjectDetectorNode()
    rospy.spin()
