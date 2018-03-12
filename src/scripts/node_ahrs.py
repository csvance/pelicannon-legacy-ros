#!/usr/bin/env python

import math
import rospy
from pelicannon.msg import XYZ, AHRS, NineDoF, NineDoFs


def xyz(t):
    return XYZ(x=t[0], y=t[1], z=t[2])


class AHRSNode(object):
    def __init__(self):
        self._ros_init()

    def _ros_init(self):
        rospy.init_node('ahrs')
        self._publisher = rospy.Publisher('heading', AHRS, queue_size=10)

        rospy.Subscriber("euler_angles", NineDoFs, self._ninedof_callback)

    def _ninedof_callback(self, ninedofs):

        for ninedof in ninedofs.ninedofs:
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

            rospy.loginfo(rospy.get_caller_id() + "Pitch %.2f Roll %.2f Yaw %.2f" % (pitch, roll, yaw))

            self._publisher.publish(AHRS(pitch=pitch, roll=roll, yaw=yaw)
)


if __name__ == "__main__":
    node = AHRSNode()
    rospy.spin()