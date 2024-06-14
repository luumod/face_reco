import cv2 as cv
from PyQt5.QtGui import QImage


def Mat2QImage(frame: cv.Mat) -> QImage:
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    height, width, depth = rgb_img.shape
    bytes_per_line = depth * width
    mage = QImage(rgb_img.data, width, height, bytes_per_line,
                  QImage.Format_RGB888)
    return mage
