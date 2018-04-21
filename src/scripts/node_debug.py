#!/usr/bin/env python

import datetime
import json
from cv_bridge import CvBridge
from threading import Lock

import cv2
import rospy
from pelicannon.msg import CategorizedRegionsOfInterest, AngleDeltaImage
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
        if rospy.get_param('object_detector/debug'):
            rospy.Subscriber('angle_delta_image', AngleDeltaImage, self._adi_callback)

        self._publisher = rospy.Publisher('cv_debug', Image, queue_size=10)

    def _adi_callback(self, adi):

        if abs(adi.last_imu[-1].angular_velocity.z) < rospy.get_param('debug/adi_min_rad_velocity'):
            return

        filename_prefix = "%d.%d" % (adi.image_initial.header.stamp.secs,
                                     adi.image_initial.header.stamp.nsecs)

        filename_i = "%s_i.png" % filename_prefix
        filename_f = "%s_f.png" % filename_prefix
        filename_d = "%s_d.json" % filename_prefix

        path_i = "%s/%s" % (rospy.get_param('debug/adi_dump_dir'), filename_i)
        path_f = "%s/%s" % (rospy.get_param('debug/adi_dump_dir'), filename_f)
        path_d = "%s/%s" % (rospy.get_param('debug/adi_dump_dir'), filename_d)

        cv2.imwrite(path_i, self._cv_br.imgmsg_to_cv2(adi.image_initial, desired_encoding='passthrough'))
        cv2.imwrite(path_f, self._cv_br.imgmsg_to_cv2(adi.image_final, desired_encoding='passthrough'))

        dat_dict = {'imu': [],
                    'image_initial': {'filename': filename_i, 'secs': adi.image_initial.header.stamp.secs,
                                      'nsecs': adi.image_initial.header.stamp.nsecs},
                    'image_final': {'filename': filename_f, 'secs': adi.image_final.header.stamp.secs,
                                    'nsecs': adi.image_final.header.stamp.nsecs}}
        for i in adi.last_imu:
            imu_record = {'orientation':
                              {'x': i.orientation.x,
                               'y': i.orientation.y,
                               'z': i.orientation.z,
                               'w': i.orientation.w},
                          'angular_velocity':
                              {'x': i.angular_velocity.x,
                               'y': i.angular_velocity.y,
                               'z': i.angular_velocity.z},
                          'linear_acceleration':
                              {'x': i.linear_acceleration.x,
                               'y': i.linear_acceleration.y,
                               'z': i.linear_acceleration.z},
                          'stamp': {'secs': i.header.stamp.secs, 'nsecs': i.header.stamp.nsecs}}
            dat_dict['imu'].append(imu_record)

        dat_file = open(path_d, 'w')
        dat_file.write(json.dumps(dat_dict))
        dat_file.close()

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
