#!/usr/bin/env python

import math

import rospy
from geometry_msgs.msg import Vector3
from pelicannon.msg import NineDoFs


def xyz(t):
    return XYZ(x=t[0], y=t[1], z=t[2])


class AHRSNode(object):
    def __init__(self):
        self._ros_init()

        self.last_pitch = None
        self.last_roll = None
        self.last_yaw = None

    def _ros_init(self):
        rospy.init_node('ahrs')
        self._publisher_ahrs = rospy.Publisher('euler_angles', Vector3, queue_size=10)
        self._publisher_angular_velocity = rospy.Publisher('angular_velocity', Vector3, queue_size=10)

        rospy.Subscriber("ninedofs", NineDoFs, self._ninedof_callback)

    def _ninedof_callback(self, ninedofs):

        delta_pitch = 0.
        delta_roll = 0.
        delta_yaw = 0.

        for ninedof_index, ninedof in enumerate(ninedofs.ninedofs):
            m = ninedof.magnometer
            a = ninedof.accelerometer

            roll = math.atan2(a.y, a.z)

            if a.y * math.sin(roll) + a.z * math.cos(roll) == 0.:
                pitch = math.pi / 2 if a.x > 0. else -math.pi / 2
            else:
                pitch = math.atan(-a.x / (a.y * math.sin(roll) + (a.z * math.cos(roll))))

            yaw = math.atan2(m.z * math.sin(roll) - m.y * math.cos(roll),
                             m.x * math.cos(pitch) +
                             m.y * math.sin(pitch) * math.sin(roll) +
                             m.z * math.sin(pitch) * math.cos(roll))

            self._publisher_ahrs.publish(Vector3(x=pitch, y=roll, z=yaw))

            if self.last_pitch is not None:
                delta_pitch += pitch - self.last_pitch
                delta_roll += roll - self.last_roll
                delta_yaw += yaw - self.last_yaw

            self.last_pitch = pitch
            self.last_roll = roll
            self.last_yaw = yaw

        av_pitch = delta_pitch / ninedofs.delta_t
        av_roll = delta_roll / ninedofs.delta_t
        av_yaw = delta_yaw / ninedofs.delta_t

        self._publisher_angular_velocity.publish(Vector3(x=av_pitch, y=av_roll, z=av_yaw))


if __name__ == "__main__":
    node = AHRSNode()
    rospy.spin()
