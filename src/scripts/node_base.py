#!/usr/bin/env python

from collections import deque

import rospy
import tf
from sensor_msgs.msg import Imu


class BaseNode(object):
    def __init__(self, integrate_num_samples=10):
        self._tf_broadcaster = None

        self._ros_init()

        self._integrate_num_samples = integrate_num_samples
        self._imu_queue = [deque(maxlen=integrate_num_samples),
                           deque(maxlen=integrate_num_samples),
                           deque(maxlen=integrate_num_samples)]
        self._base_position = [0., 0., 0.]

    def _ros_init(self):
        rospy.init_node('base')
        rospy.Subscriber("/imu/data", Imu, self._imu_callback)
        self._publisher_imu = rospy.Publisher('/imu/data_base', Imu, queue_size=10)

        self._tf_broadcaster = tf.TransformBroadcaster()

    # 100hz
    def _imu_callback(self, imu):
        self._tf_broadcaster.sendTransform((self._base_position[0],
                                            self._base_position[1],
                                            self._base_position[2]),
                                           (0., 0., 0., 1.), imu.header.stamp,
                                           "base", "world")
        self._tf_broadcaster.sendTransform((0., 0., rospy.get_param('base/camera_height')),
                                           (imu.orientation.x,
                                            imu.orientation.y,
                                            imu.orientation.z,
                                            imu.orientation.w),
                                           imu.header.stamp,
                                           "webcam", "base")
        self._tf_broadcaster.sendTransform((0., 0., 0.), (0., 0., 0., 1.), imu.header.stamp,
                                           "nerf", "webcam")

        imu.header.frame_id = "webcam"
        imu.header.stamp = rospy.Time.from_sec(rospy.get_time())
        self._publisher_imu.publish(imu)


if __name__ == '__main__':
    pose = BaseNode()
    rospy.spin()
