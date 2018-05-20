#!/usr/bin/env python

import datetime
from collections import deque
from cv_bridge import CvBridge
from threading import Lock

import cv2
import numpy as np
import rospy
from pelicannon.msg import CategorizedRegionOfInterest, CategorizedRegionsOfInterest
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32


class Rectangle(object):
    def __init__(self, rect_tuple, scale=(1., 1.)):
        x, y, w, h = rect_tuple
        self.x = int(x * scale[0])
        self.y = int(y * scale[1])
        self.w = int(w * scale[0])
        self.h = int(h * scale[1])

    def scale(self, scale_x, scale_y):
        x = int(scale_x * self.x)
        w = int(scale_x * self.x)
        y = int(scale_y * self.y)
        h = int(scale_y * self.h)
        return Rectangle((x, y, w, h))

    def __repr__(self):
        return "x: %d w: %d y: %d h: %d" % (self.x, self.w, self.y, self.h)


class BodyTrackerPipeline(object):
    def __init__(self, min_x=0, min_y=0):
        self._cv_br = CvBridge()

        self._publisher_image_debug = rospy.Publisher('cv_peek', Image, queue_size=1)

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self._min_x = min_x
        self._min_y = min_y

    def process_frame(self, frame):
        # Workaround scale invariance issue in OpenCV HOG-SVM by vertical padding
        combined = np.pad(frame.copy(), ((0, frame.shape[0]),
                                         (0, 0), (0, 0)),
                          mode='constant', constant_values=0)

        rectangles = []
        rects, weights = self._hog.detectMultiScale(combined, scale=1.05)
        for (x, y, w, h) in rects:
            h /= 2
            rectangles.append(Rectangle((x, y, w, h)))

        return rectangles


class MotionTrackerPipeline(object):
    def __init__(self, coeff_min_area=0.025, coeff_max_area=0.75, in_motion=False):
        self.frame_initial = None
        self.coeff_min_area = coeff_min_area
        self.coeff_max_area = coeff_max_area

        self._phi_history = deque(maxlen=8)

        self._cv_br = CvBridge()
        self._publisher_image_debug = rospy.Publisher('cv_peek', Image, queue_size=1)

    def process_frame(self, frame, phi=None):

        self._phi_history.append(phi if phi is not None else 0.0)

        if self.frame_initial is None:
            self.frame_initial = frame
            return []

        frame_initial = self.frame_initial
        frame_final = frame

        matrix_hist = np.array(self._phi_history)
        if np.max(np.abs(matrix_hist)) > 0.01:
            return []

        frame_delta = cv2.absdiff(frame_initial, frame_final)

        if rospy.get_param('debug/video_source') == '/pelicannon/image_abs_diff':
            self._publisher_image_debug.publish(self._cv_br.cv2_to_imgmsg(frame_delta, encoding="passthrough"))

        thresh = cv2.threshold(frame_delta, 24, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=3)

        if rospy.get_param('debug/video_source') == '/pelicannon/image_thresh':
            self._publisher_image_debug.publish(self._cv_br.cv2_to_imgmsg(thresh, encoding="passthrough"))

        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < int(self.coeff_min_area * frame.shape[0] * frame.shape[1]) or area > int(
                    self.coeff_max_area * frame.shape[0] * frame.shape[1]):
                continue
            rectangles.append(Rectangle(cv2.boundingRect(c)))

        # Store this frame as the next frame initial
        self.frame_initial = frame_final

        return rectangles


class ObjectDetectorNode(object):
    def __init__(self, debug=False):
        self._debug = debug
        self._initialize_pipelines()
        self._cv_br = CvBridge()

        self._ahrs_lock = Lock()
        self._ahrs = None

        self._imu_lock = Lock()
        self._imu_queue = deque(maxlen=3)

        self._last_image = None

        self._ros_init()

    def _ros_init(self):
        rospy.init_node('object_detector')
        self._publisher = rospy.Publisher('regions_of_interest', CategorizedRegionsOfInterest, queue_size=1)
        self._publisher_move = rospy.Publisher('/motor/move_angle', Float32, queue_size=1)
        self._publisher_image_debug = rospy.Publisher('cv_peek', Image, queue_size=1)

        rospy.Subscriber("/webcam/image_raw", Image, self._camera_callback)
        rospy.Subscriber("/imu/data", Imu, self._imu_callback)

    def _initialize_pipelines(self):
        if rospy.get_param('object_detector/body_regions'):
            self._body_tracker = BodyTrackerPipeline()
        if rospy.get_param('object_detector/motion_regions'):
            self._motion_detector = MotionTrackerPipeline()

    def _imu_callback(self, imu):
        self._imu_lock.acquire()
        self._imu_queue.append(imu)
        self._imu_lock.release()

    def _compute_angular_velocity(self):
        a_x = 0.
        a_y = 0.
        a_z = 0.

        self._imu_lock.acquire()

        if len(self._imu_queue) == 0:
            self._imu_lock.release()
            return a_x, a_y, a_z

        rad_velocity = np.zeros((len(self._imu_queue), 3))

        for idx, i in enumerate(self._imu_queue):
            rad_velocity[idx][0] = i.angular_velocity.x
            rad_velocity[idx][1] = i.angular_velocity.y
            rad_velocity[idx][2] = i.angular_velocity.z

        self._imu_lock.release()

        return np.max(rad_velocity[:,0]), np.max(rad_velocity[:,1]), np.max(rad_velocity[:,2])

    def _camera_callback(self, image):

        delta_t = (image.header.stamp - self._last_image.header.stamp).to_sec() if self._last_image is not None else 0.

        time_i = datetime.datetime.now()

        frame = self._cv_br.imgmsg_to_cv2(image, desired_encoding="passthrough")

        if rospy.get_param('debug/video_source') == '/pelicannon/image_raw':
            self._publisher_image_debug.publish(self._cv_br.cv2_to_imgmsg(frame, encoding="passthrough"))

        # Create grayscale versions and equalize
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_grayscale = cv2.equalizeHist(frame_grayscale)

        av_x, av_y, av_z = self._compute_angular_velocity()

        # Send data down pipeline and process results
        regions = []

        if rospy.get_param('object_detector/body_regions'):
            haarcascade_regions = self._body_tracker.process_frame(frame)
            for region in haarcascade_regions:
                regions.append(
                    CategorizedRegionOfInterest(x=region.x, y=region.y, w=region.w, h=region.h, category='body'))

        if rospy.get_param('object_detector/motion_regions'):
            motion_regions = self._motion_detector.process_frame(frame_grayscale,
                                                                 phi=av_z * delta_t
                                                                 if len(self._imu_queue) != 0 and
                                                                    abs(av_z * delta_t) > 0.01 else None)
            for region in motion_regions:
                regions.append(
                    CategorizedRegionOfInterest(x=region.x, y=region.y, w=region.w, h=region.h, category='motion'))

        self._publisher.publish(CategorizedRegionsOfInterest(regions=regions))

        # Give priority to body regions
        regions.sort(key=lambda k: k.category, reverse=False)
        for region in regions:

            region_x_midpoint = (region.x + region.x + region.w) / 2.
            frame_midpoint = frame.shape[1] / 2

            delta_midpoint = region_x_midpoint - frame_midpoint

            angle_coeff = delta_midpoint / frame_midpoint
            rot_angle = angle_coeff * 78 * np.pi / 180.

            if region.category == 'body':
                self._publisher_move.publish(rot_angle / 4.)
            elif region.category == 'motion':
                self._publisher_move.publish(rot_angle)
            break

        time_f = datetime.datetime.now()

        if self._debug:
            rospy.loginfo(rospy.get_caller_id() + "Frame processed in %s", time_f - time_i)

        self._last_image = image


if __name__ == '__main__':
    detector = ObjectDetectorNode()
    rospy.spin()
