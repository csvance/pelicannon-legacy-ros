#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3
from pelicannon.msg import NineDoFs, NineDoF

from lsm9ds0 import LSM9DS0


class LSM9DS0Node(object):
    def __init__(self):
        self._ros_init()

        self._sensor = LSM9DS0(callback=self._sensor_callback, i2c_bus_num=1, fifo_size=8)
        self._sensor.start()

    def _ros_init(self):
        rospy.init_node('lsm9ds0')
        self._publisher = rospy.Publisher('ninedofs', NineDoFs, queue_size=10)

    def _sensor_callback(self, accelerometer, magnometer, gyrometer):
        ninedofs = []
        for i in range(0, len(accelerometer)):
            a = Vector3(x=accelerometer[i][0], y=accelerometer[i][1], z=accelerometer[i][2])
            m = Vector3(x=magnometer[i][0], y=magnometer[i][1], z=magnometer[i][2])
            g = Vector3(x=gyrometer[i][0], y=gyrometer[i][1], z=gyrometer[i][2])

            ninedofs.append(NineDoF(accelerometer=a, magnometer=m, gyrometer=g))

        self._publisher.publish(NineDoFs(ninedofs=ninedofs, delta_t=len(accelerometer) / 100.))

    def shutdown(self):
        self._sensor.shutdown()


if __name__ == "__main__":
    node = LSM9DS0Node()
    rospy.spin()
    node.shutdown()
