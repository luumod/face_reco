from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
import cv2 as cv
import numpy as np


class CameraCapture(QThread):
    frame_signal = pyqtSignal(np.ndarray, int)
    frame_signal_detect = pyqtSignal(np.ndarray, int)

    def __init__(self, w, h):
        super(QThread, self).__init__()
        self.cap = cv.VideoCapture(0)
        self.mutex = QMutex()
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
        self.FPS = self.cap.get(cv.CAP_PROP_FPS)
        self.is_running = False

    def run(self):
        while self.is_running:
            self.mutex.lock()
            if not self.cap.isOpened():
                self.mutex.unlock()
                break
            flag, frame = self.cap.read()
            if not flag:
                self.mutex.unlock()
                break
            self.frame_signal.emit(frame, 0)
            # 处理图片操作
            self.frame_signal_detect.emit(frame, 1)

            cv.waitKey(int(1000 / self.FPS))
            self.mutex.unlock()

    def change_choice(self):
        self.choice = 1

    def quit(self):
        self.cap.release()
        cv.destroyAllWindows()
