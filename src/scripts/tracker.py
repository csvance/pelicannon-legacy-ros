import cv2
import numpy as np
from image_transformer import ImageTransformer
import skimage.measure


class Rectangle(object):
    def __init__(self, rect_tuple, scale=(1., 1.)):
        x, y, w, h = rect_tuple
        self.x = int(x * scale[0])
        self.y = int(y * scale[1])
        self.w = int(w * scale[0])
        self.h = int(h * scale[1])

    def scale(self, scale_x, scale_y):
        x = int(scale_x * self.x)
        w = int(scale_x * self.x)
        y = int(scale_y * self.y)
        h = int(scale_y * self.h)
        return Rectangle((x, y, w, h))

    def __repr__(self):
        return "x: %d w: %d y: %d h: %d" % (self.x, self.w, self.y, self.h)


class BodyTrackerPipeline(object):
    def __init__(self, min_x=0, min_y=0):
        self._cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_fullbody.xml')
        self._min_x = min_x
        self._min_y = min_y

    def process_frame(self, frame):
        rectangles = []
        for r in self._cascade.detectMultiScale(frame,
                                                1.05, 1, 0,
                                                (self._min_x, self._min_y)):
            rectangles.append(Rectangle(r))
        return rectangles


class MotionTrackerPipeline(object):
    def __init__(self, coeff_min_area=0.025, coeff_max_area=1.0):
        self.frame_initial = None
        self.coeff_min_area = coeff_min_area
        self.coeff_max_area = coeff_max_area

    def process_frame(self, frame, phi=None, rescale=(1., 1.)):

        if self.frame_initial is None:
            self.frame_initial = frame
            return []

        frame_initial = self.frame_initial
        frame_final = frame

        if phi is not None:
            frame_initial_warped = np.reshape(frame_initial.copy(), (frame.shape[0], frame.shape[1], 1))

            # Needs to be calibrated more
            dx = int(-2. / 15. * phi)

            it = ImageTransformer(frame_initial_warped)
            frame_initial_warped = it.rotate_along_axis(phi=phi, dx=dx)
            frame_initial = frame_initial_warped

        frame_delta = cv2.absdiff(frame_initial, frame_final)

        if phi is not None:
            # zero things based on our dx shifting
            if dx > 0:
                frame_delta[0:rescale[1], 0:dx] = 0
            elif dx < 0:
                frame_delta[0:rescale[1], dx:rescale[1]] = 0

        pool = skimage.measure.block_reduce(frame_delta, (2, 2), np.average).astype(np.uint8)

        pool_scale_x = frame.shape[0] / pool.shape[0]
        pool_scale_y = frame.shape[1] / pool.shape[1]

        thresh = cv2.threshold(pool, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)

        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < int(self.coeff_min_area*pool.shape[0]*pool.shape[1]) or area > int(self.coeff_max_area*pool.shape[0]*pool.shape[1]):
                continue
            rectangles.append(Rectangle(cv2.boundingRect(c), scale=(pool_scale_x, pool_scale_y)))

        # Store this frame as the next frame initial
        self.frame_initial = frame_final
        print(rectangles)
        return rectangles

