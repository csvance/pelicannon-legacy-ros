#!/usr/bin/env python

import StringIO
import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
from cv_bridge import CvBridge
from threading import Lock
import signal
import socket
import datetime

import Image as PyImage
import cv2
import rospy
from pelicannon.msg import CategorizedRegionsOfInterest
from sensor_msgs.msg import Image
from Queue import Queue

image_queue = Queue(maxsize=1)


class FrameHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    img = image_queue.get(timeout=1)

                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    jpg = PyImage.fromarray(imgRGB)
                    tmpFile = StringIO.StringIO()
                    jpg.save(tmpFile, 'JPEG')
                    self.wfile.write("--jpgboundary")
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(tmpFile.len))
                    self.end_headers()
                    jpg.save(self.wfile, 'JPEG')
                    time.sleep(0.05)
                except KeyboardInterrupt:
                    break
            return
        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img src="http://127.0.0.1:8080/cam.mjpg"/>')
            self.wfile.write('</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


class DebugNode(object):
    def __init__(self, target_fps=30):
        self._cv_br = CvBridge()

        self._ros_init()

        self._roi_lock = Lock()
        self._latest_rois = []

        self._target_fps = target_fps
        self._frames_processed = 0
        self._start_time = datetime.datetime.now()

        self._server = ThreadedHTTPServer(('0.0.0.0', 8080), FrameHandler)

    def _ros_init(self):
        rospy.init_node('debug')

        rospy.Subscriber('regions_of_interest', CategorizedRegionsOfInterest, self._roi_callback)
        rospy.Subscriber("/webcam/image_raw", Image, self._camera_callback)

    def _roi_callback(self, rois):

        self._roi_lock.acquire()
        self._latest_rois = rois.regions
        self._roi_lock.release()

    def _camera_callback(self, image):
        global image_queue, image_event

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

        with image_queue.mutex:
            image_queue.queue.clear()
        image_queue.put(frame)

        self._frames_processed += 1

    def run(self):
        try:
            self._server.serve_forever()
        except socket.error as e:
            # Mute the invalid file descriptor exception when shutting down
            if e.errno == 9:
                pass
            else:
                raise e

    def shutdown(self):
        self._server.socket.close()


def stop_signal_callback(signo, stack_frame):
        node.shutdown()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, stop_signal_callback)
    signal.signal(signal.SIGTERM, stop_signal_callback)

    node = DebugNode()
    node.run()
