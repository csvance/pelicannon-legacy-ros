#!/usr/bin/env python

import struct
from threading import Lock

import numpy as np
import rospy
import serial
from sensor_msgs.msg import Imu, MagneticField
from std_msgs.msg import Int16


class K66FNode(object):
    def __init__(self):
        self._ros_init()

        self._ser = serial.Serial(rospy.get_param('k66f/tty'), int(rospy.get_param('k66f/baud')))
        self._ser_lock = Lock()
        self._gyro_cal = {'x': rospy.get_param('k66f/gyro_cal_x'),
                          'y': rospy.get_param('k66f/gyro_cal_y'),
                          'z': rospy.get_param('k66f/gyro_cal_z')}

    def _ros_init(self):
        rospy.init_node('k66f')
        self._publisher_imu = rospy.Publisher('/imu/data_raw', Imu, queue_size=10)
        self._publisher_magnetic = rospy.Publisher('/imu/mag', MagneticField, queue_size=10)

        rospy.Subscriber("/motor/direction", Int16, self._motor_callback)

    def _motor_callback(self, steps):
        self._ser_lock.acquire()

        if steps == 0:
            self._ser.write('A;')
        else:
            self._ser.write('S:%d;' % steps)

        self._ser_lock.release()

    def _synchronize(self):
        rospy.loginfo(rospy.get_caller_id() + " Resynchronizing stream...")

        self._ser_lock.acquire()

        while True:

            b1 = self._ser.read(1)
            if b1 != "\xDE":
                continue
            b2 = self._ser.read(1)
            if b2 != "\xAD":
                continue
            b3 = self._ser.read(1)
            if b3 != "\xBE":
                continue
            b4 = self._ser.read(1)
            if b4 != "\xEF":
                continue

            self._ser.read(18)
            break

        self._ser_lock.release()

    def _calibrate(self, samples=4096):
        rospy.loginfo(rospy.get_caller_id() + " Running calibration...")

        self._synchronize()

        self._gyro_cal['x'] = 0.
        self._gyro_cal['y'] = 0.
        self._gyro_cal['z'] = 0.

        data = np.zeros((samples, 3))

        for i in range(0, samples):
            imu, magnetic = self._read()

            data[i][0] = imu.angular_velocity.x
            data[i][1] = imu.angular_velocity.y
            data[i][2] = imu.angular_velocity.z

        x, y, z = np.average(data[:, 0]), np.average(data[:, 1]), np.average(data[:, 2])
        print("Gyro Calibration Complete - X: %f Y: %f Z: %f" % (x, y, z))

        self._gyro_cal['x'] = x
        self._gyro_cal['y'] = y
        self._gyro_cal['z'] = z

    def _read(self):
        self._ser_lock.acquire()
        data = self._ser.read(22)
        self._ser_lock.release()

        ts = rospy.get_time()

        if data[0] != "\xDE" or data[1] != "\xAD" or data[2] != "\xBE" or data[3] != "\xEF":
            return False

        imu = Imu()
        magnetic = MagneticField()

        stamp = rospy.Time.from_sec(ts)

        imu.header.stamp = stamp
        magnetic.header.stamp = stamp

        imu.orientation_covariance[0] = -1.

        imu.angular_velocity.x = struct.unpack("h", data[4:6])[0] * np.pi / 180.
        imu.angular_velocity.y = struct.unpack("h", data[6:8])[0] * np.pi / 180.
        imu.angular_velocity.z = struct.unpack("h", data[8:10])[0] * np.pi / 180.

        # Apply Calibration
        imu.angular_velocity.x += self._gyro_cal['x']
        imu.angular_velocity.y += self._gyro_cal['y']
        imu.angular_velocity.z += self._gyro_cal['z']

        imu.angular_velocity_covariance[0] = -1.

        imu.linear_acceleration.x = struct.unpack("h", data[10:12])[0]
        imu.linear_acceleration.y = struct.unpack("h", data[12:14])[0]
        imu.linear_acceleration.z = struct.unpack("h", data[14:16])[0]
        imu.linear_acceleration_covariance[0] = -1.

        magnetic.magnetic_field.x = struct.unpack("h", data[16:18])[0]
        magnetic.magnetic_field.y = struct.unpack("h", data[18:20])[0]
        magnetic.magnetic_field.z = struct.unpack("h", data[20:22])[0]

        return imu, magnetic

    def run(self):
        self._synchronize()

        if rospy.get_param('k66f/run_calibration'):
            self._calibrate()

        while not rospy.is_shutdown():

            t = self._read()
            if t is False:
                self._synchronize()
            # Unpack
            try:
                imu, magnetic = t
            except TypeError:
                self._synchronize()
                continue
            self._publisher_imu.publish(imu)
            self._publisher_magnetic.publish(magnetic)

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


if __name__ == "__main__":
    node = K66FNode()
    node.run()
