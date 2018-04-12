#!/usr/bin/env python

import math
from collections import deque

import rospy
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion


class VelocityNode(object):
    def __init__(self):
        self._ros_init()

        self._euler_que = deque(maxlen=8)
        self._timestamp_queue = deque(maxlen=8)

    def _ros_init(self):
        rospy.init_node('velocity')
        self._publisher_angular_velocity = rospy.Publisher('angular_velocity', Vector3, queue_size=10)

        rospy.Subscriber("/imu/data", Imu, self._imu_callback)

    def _imu_callback(self, imu):

        self._euler_que.append(euler_from_quaternion([imu.orientation.x,
                                                      imu.orientation.y,
                                                      imu.orientation.z,
                                                      imu.orientation.w]))
        self._timestamp_queue.append(imu.header.stamp.to_sec())

        d_pitch = 0.
        d_roll = 0.
        d_yaw = 0.
        d_time = 0.

        last_pitch = None
        last_roll = None
        last_yaw = None
        last_time = None

        def angle_diff(t1, t2):
            if abs(t2 - t1) > math.pi:
                if t2 > t1:
                    t1 += 2 * math.pi
                elif t1 > t2:
                    t2 += 2 * math.pi

            return t2 - t1

        for idx, (roll, pitch, yaw) in enumerate(self._euler_que):
            if last_pitch is not None:
                d_roll += angle_diff(roll, last_roll)
                d_pitch += angle_diff(pitch, last_pitch)
                d_yaw += angle_diff(yaw, last_yaw)
                d_time += self._timestamp_queue[idx] - last_time

            last_roll = roll
            last_pitch = pitch
            last_yaw = yaw
            last_time = self._timestamp_queue[idx]

        if d_time > 0.:
            av_pitch = d_pitch / d_time
            av_roll = d_roll / d_time
            av_yaw = d_yaw / d_time

            self._publisher_angular_velocity.publish(Vector3(x=av_roll, y=av_pitch, z=av_yaw))


if __name__ == "__main__":
    node = VelocityNode()
    rospy.spin()
