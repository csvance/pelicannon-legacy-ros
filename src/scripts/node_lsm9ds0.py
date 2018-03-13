#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu, MagneticField

from lsm9ds0 import LSM9DS0


class LSM9DS0Node(object):
    def __init__(self):
        self._ros_init()

        self._sensor = LSM9DS0(callback=self._sensor_callback, i2c_bus_num=1, fifo_size=1)
        self._sensor.start()

    def _ros_init(self):
        rospy.init_node('lsm9ds0')
        self._publisher_imu = rospy.Publisher('/imu/data_raw', Imu, queue_size=10)
        self._publisher_magnetic = rospy.Publisher('/imu/mag', MagneticField, queue_size=10)

    def _sensor_callback(self, accelerometer, magnometer, gyrometer):
        end_ts = rospy.get_time()

        for i in range(0, len(accelerometer)):
            imu = Imu()
            magnetic = MagneticField()

            # Calculate timestamp for reading
            samples_behind = (len(accelerometer) - 1) - i
            samples_per_sec = len(accelerometer) / 50.
            stamp = rospy.Time.from_sec(end_ts - samples_behind * samples_per_sec)

            imu.header.stamp = stamp
            magnetic.header.stamp = stamp

            imu.orientation_covariance[0] = -1.

            imu.linear_acceleration.x = accelerometer[i][0]
            imu.linear_acceleration.y = accelerometer[i][1]
            imu.linear_acceleration.z = accelerometer[i][2]
            imu.linear_acceleration_covariance[0] = -1.

            imu.angular_velocity.x = gyrometer[i][0]
            imu.angular_velocity.y = gyrometer[i][1]
            imu.angular_velocity.z = gyrometer[i][2]
            imu.angular_velocity_covariance[0] = -1.

            magnetic.magnetic_field.x = magnometer[i][0]
            magnetic.magnetic_field.y = magnometer[i][1]
            magnetic.magnetic_field.z = magnometer[i][2]

            self._publisher_imu.publish(imu)
            self._publisher_magnetic.publish(magnetic)

    def shutdown(self):
        self._sensor.shutdown()


if __name__ == "__main__":
    node = LSM9DS0Node()
    rospy.spin()
    node.shutdown()
