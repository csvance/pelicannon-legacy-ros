#!/usr/bin/env python

import cv2
from tracker import BodyTrackerPipeline, MotionTrackerPipeline
import datetime
from cv_bridge import CvBridge

import rospy
from sensor_msgs.msg import Image
from pelicannon.msg import CategorizedRegionOfInterest, CategorizedRegionsOfInterest, AHRS


class ObjectDetectorNode(object):
    def __init__(self, debug=False):
        self._debug = debug
        self._initialize_pipelines()
        self._cv_br = CvBridge()

        self.phi = None

        self._ros_init()

    def _ros_init(self):
        rospy.init_node('object_detector')
        self._publisher = rospy.Publisher('object_segments', CategorizedRegionsOfInterest, queue_size=10)

        rospy.Subscriber("/camera/image_raw", Image, self._camera_callback)
        rospy.Subscriber("euler_angles", AHRS, self._ahrs_callback)

    def _initialize_pipelines(self):
        self._body_tracker = BodyTrackerPipeline()
        self._motion_detector = MotionTrackerPipeline()

        if self._debug:
            self._video_writer = cv2.VideoWriter('output.avi', cv2.cv.FOURCC('M', 'J', 'P', 'G'), 30.0, (160, 90))
        else:
            self._video_writer = None

    def _ahrs_callback(self, ninedof):
        pass

    def _camera_callback(self, image):

        time_i = datetime.datetime.now()

        frame = self._cv_br.imgmsg_to_cv2(image, desired_encoding="passthrough")
        frame_original = frame.copy()

        # Create grayscale versions
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Does this improve performance? In example code in OpenCV documentation.
        frame_grayscale = cv2.equalizeHist(frame_grayscale)

        # Send data down pipeline and process results
        haarcascade_regions = self._body_tracker.process_frame(frame_grayscale)
        motion_regions = self._motion_detector.process_frame(frame_grayscale, phi=self.phi)

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
