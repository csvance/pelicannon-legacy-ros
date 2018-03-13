#!/usr/bin/env python

from collections import deque

import rospy
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion


class AHRSNode(object):
    def __init__(self):
        self._ros_init()

        self._euler_que = deque(maxlen=32)
        self._timestamp_queue = deque(maxlen=32)

    def _ros_init(self):
        rospy.init_node('ahrs')
        self._publisher_ahrs = rospy.Publisher('euler_angles', Vector3, queue_size=10)
        self._publisher_angular_velocity = rospy.Publisher('angular_velocity', Vector3, queue_size=10)

        rospy.Subscriber("/imu/data", Imu, self._imu_callback)

    def _imu_callback(self, imu):

        self._euler_que.append(euler_from_quaternion([imu.orientation.x,
                                                      imu.orientation.y,
                                                      imu.orientation.z,
                                                      imu.orientation.w], 'sxyz'))
        self._timestamp_queue.append(imu.header.stamp.to_sec())

        print("roll(%.2f) pitch(%.2f) yaw(%.2f)" % (
        self._euler_que[-1][0], self._euler_que[-1][1], self._euler_que[-1][2]))

        delta_pitch = 0.
        delta_roll = 0.
        delta_yaw = 0.
        delta_time = 0.

        last_pitch = None
        last_roll = None
        last_yaw = None
        last_time = None

        for idx, (roll, pitch, yaw) in enumerate(self._euler_que):
            if last_pitch is not None:
                delta_roll += roll - last_roll
                delta_pitch += pitch - last_pitch
                delta_yaw += yaw - last_yaw
                delta_time += self._timestamp_queue[idx] - last_time

            last_roll = roll
            last_pitch = pitch
            last_yaw = yaw
            last_time = self._timestamp_queue[idx]

        if delta_time > 0.:
            av_pitch = delta_pitch / delta_time
            av_roll = delta_roll / delta_time
            av_yaw = delta_yaw / delta_time
            self._publisher_angular_velocity.publish(Vector3(x=av_roll, y=av_pitch, z=av_yaw))
        else:
            print delta_time


if __name__ == "__main__":
    node = AHRSNode()
    rospy.spin()
