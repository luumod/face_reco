import threading
import time
import face_recognition
import numpy as np
from PyQt5.QtCore import QMutex, pyqtSignal
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPainter, QPaintEvent
import cv2 as cv
from mat2qimage import Mat2QImage
from imutils import face_utils
import dlib
import os


class VideoWidget(QLabel):
    send_detect_area_signal = pyqtSignal(np.ndarray, int)
    send_image_1 = pyqtSignal(np.ndarray)
    send_image_2 = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(VideoWidget, self).__init__(parent)
        self.dst_image = QImage()
        self.use_mask = False
        # 人脸  轮廓  眼睛  鼻子 嘴巴
        self.cbx_id = 0
        self.save_idx = [1, 1, 1, 1, 1]
        self.mask_image = cv.imread('assets/mask1.png')
        self.is_save = False
        self.save_path = "./save"
        self.save_thread = None
        self.save_lock = threading.Lock()
        self.is_thread_begin = False
        self.mutex = QMutex()
        self.use_video = False
        self.frame_width = 0
        self.frame_height = 0
        self.face_detector = cv.CascadeClassifier(
            'F:/Tools/openCV/openCV/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
        self.eye_detector = cv.CascadeClassifier('F:/Tools/openCV/openCV/sources/data/haarcascades/haarcascade_eye.xml')
        self.nose_detector = cv.CascadeClassifier(
            'F:/Tools/openCV/openCV/sources/data/haarcascades/haarcascade_mcs_nose.xml')
        self.mouth_detector = cv.CascadeClassifier(
            'F:/Tools/openCV/openCV/sources/data/haarcascades/haarcascade_mcs_mouth.xml')

        # 轮廓检测
        self.contours_p = "data/shape_predictor_68_face_landmarks.dat"
        self.contours_detector = dlib.get_frontal_face_detector()
        self.contours_predictor = dlib.shape_predictor(self.contours_p)

    # 加载静态图片
    def load_image_1(self, mt_1):
        image_width, image_height = mt_1.shape[:2]
        www = 480 / image_width
        hhh = 480 / image_height
        scale_factor = min(www, hhh)
        t_mt_resized_1 = cv.resize(mt_1, (int(image_height * scale_factor), int(image_width * scale_factor)))
        # 发射给第二个
        self.send_image_1.emit(t_mt_resized_1)
        self.dst_image = Mat2QImage(t_mt_resized_1)
        self.update()

    def load_image_2(self, mat):
        if self.cbx_id == 0:
            single_image_2, single_image_3 = self.draw_face_rectangle(mat.copy())
        elif self.cbx_id == 1:
            single_image_2, single_image_3 = self.draw_face_rectangle(mat.copy())
        elif self.cbx_id == 2:
            single_image_2, single_image_3 = self.draw_eyes_rectangle(mat.copy())
        elif self.cbx_id == 3:
            single_image_2, single_image_3 = self.draw_nose_rectangle(mat.copy())
        elif self.cbx_id == 4:
            single_image_2, single_image_3 = self.draw_mouth_rectangle(mat.copy())

        if single_image_2 is None or single_image_3 is None:
            self.dst_image = None
            self.update()
            return

        self.send_image_2.emit(single_image_3)
        # 发送给第三个
        self.dst_image = Mat2QImage(single_image_2)
        self.update()

    def load_image_3(self, mat):
        image_width, image_height = mat.shape[:2]
        www = 200 / image_width
        hhh = 200 / image_height
        scale_factor = min(www, hhh)
        res = cv.resize(mat, (int(image_height * scale_factor), int(image_width * scale_factor)))
        self.dst_image = Mat2QImage(res)
        self.update()

    def getMat(self, mat: cv.Mat, flag: int):
        if flag == 0:
            if self.use_video:
                www = 640 / self.frame_width
                hhh = 480 / self.frame_height
                scale_factor = min(www, hhh)
                img_h = int(self.frame_width * scale_factor)
                img_w = int(self.frame_height * scale_factor)
                res = cv.resize(mat, (img_h, img_w))
                self.dst_image = Mat2QImage(res)
            else:
                self.dst_image = Mat2QImage(mat)
        elif flag == 1:
            if self.use_video:
                www = 640 / self.frame_width
                hhh = 480 / self.frame_height
                scale_factor = min(www, hhh)
                img_h = int(self.frame_width * scale_factor)
                img_w = int(self.frame_height * scale_factor)
                res = cv.resize(mat, (img_h, img_w))
                mat = res
                self.dst_image = Mat2QImage(mat)
            else:
                self.dst_image = Mat2QImage(mat)

            a, b, c = self.face_detect(mat)
            self.dst_image = Mat2QImage(a)
            if c:
                self.send_detect_area_signal.emit(b, 2)
        elif flag == 2:
            image_width, image_height = mat.shape[:2]
            www = 200 / image_width
            hhh = 200 / image_height
            scale_factor = min(www, hhh)
            res = cv.resize(mat, (int(image_height * scale_factor), int(image_width * scale_factor)))
            self.dst_image = Mat2QImage(res)
        self.update()

    def paintEvent(self, e: QPaintEvent):
        if self.dst_image is None or self.dst_image.isNull():
            return

        painter = QPainter(self)
        painter.drawImage(0, 0, self.dst_image)

    def save_image(self, img, dir: str, i: int):
        with self.save_lock:
            if not os.path.exists(dir):
                os.makedirs(dir)
            file = os.path.join(dir, f"{self.save_idx[i]}.png")
            if cv.imwrite(file, img):
                self.save_idx[i] += 1
            else:
                print(f"Failed to save image: {file}")

        # 调整保存时间
        time.sleep(0.3)

    def face_detect(self, img: cv.Mat):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        min_size = (30, 30)
        detect_area = np.empty((0,))
        ok = False

        if self.cbx_id == 0:
            # 人脸检测
            faces = self.face_detector.detectMultiScale(gray, minSize=min_size)
            for x, y, w, h in faces:
                detect_area = img.copy()[y:y + h, x:x + w]
                ok = True
                if not self.use_mask:
                    cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
                    cv.circle(img, center=(x + w // 2, y + h // 2), radius=(w // 2), color=(0, 255, 0), thickness=2)

                    # 启动线程保存图片
                    if self.is_save:
                        if self.save_thread is None or not self.save_thread.is_alive():
                            self.save_thread = threading.Thread(target=self.save_image,
                                                                args=(detect_area, "./save/face", 0))
                            self.save_thread.start()
                else:
                    mask = cv.resize(self.mask_image, dsize=(w, h))
                    roi_rect = (x, y, w, h)
                    img[roi_rect[1]:roi_rect[1] + roi_rect[3], roi_rect[0]:roi_rect[0] + roi_rect[2]] = mask
        elif self.cbx_id == 1:
            # 轮廓检测
            rects = self.contours_detector(gray, 0)
            try:
                for (i, rect) in enumerate(rects):
                    shape = self.contours_predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    detect_area = np.zeros_like(img)
                    ok = True
                    for (x, y) in shape:
                        cv.circle(detect_area, (x, y), 2, (255, 255, 255), -1)
                        cv.circle(img, (x, y), 2, (0, 255, 0), -1)
                        # 启动线程保存图片
                        if self.is_save:
                            if self.save_thread is None or not self.save_thread.is_alive():
                                self.save_thread = threading.Thread(target=self.save_image,
                                                                    args=(detect_area, "./save/contours", 1))
                                self.save_thread.start()
            except:
                ok = False
                print("d")
        elif self.cbx_id == 2:
            # 眼球检测
            faces = self.face_detector.detectMultiScale(gray, minSize=min_size)
            for x, y, w, h in faces:
                roi_face = gray[y:y + h, x:x + w]
                eyes = self.eye_detector.detectMultiScale(roi_face, minSize=min_size)
                for nx, ny, nw, nh in eyes:
                    detect_area = img.copy()[y + ny:y + ny + nh, x + nx:x + nx + nw]
                    ok = True
                    cv.rectangle(img, (x + nx, y + ny), (x + nx + nw, y + ny + nh), color=(0, 0, 255), thickness=2)

                    # 启动线程保存图片
                    if self.is_save:
                        if self.save_thread is None or not self.save_thread.is_alive():
                            self.save_thread = threading.Thread(target=self.save_image,
                                                                args=(detect_area, "./save/eyes", 2))
                            self.save_thread.start()

        elif self.cbx_id == 3:
            # 鼻子检测
            faces = self.face_detector.detectMultiScale(gray, minSize=min_size)
            for x, y, w, h in faces:
                roi_face = gray[y:y + h, x:x + w]
                nose = self.nose_detector.detectMultiScale(roi_face, 1.2, 10)
                for nx, ny, nw, nh in nose:
                    detect_area = img.copy()[y + ny:y + ny + nh, x + nx:x + nx + nw]
                    ok = True
                    cv.rectangle(img, (x + nx, y + ny), (x + nx + nw, y + ny + nh), color=(0, 0, 255), thickness=2)
                    # 启动线程保存图片
                    if self.is_save:
                        if self.save_thread is None or not self.save_thread.is_alive():
                            self.save_thread = threading.Thread(target=self.save_image,
                                                                args=(detect_area, "./save/nose", 3))
                            self.save_thread.start()
        elif self.cbx_id == 4:
            # 嘴巴检测
            faces = self.face_detector.detectMultiScale(gray, minSize=min_size)
            for x, y, w, h in faces:
                roi_face = gray[y:y + h, x:x + w]
                mouth = self.mouth_detector.detectMultiScale(roi_face, minSize=(50, 50))
                for nx, ny, nw, nh in mouth:
                    detect_area = img.copy()[y + ny:y + ny + nh, x + nx:x + nx + nw]
                    ok = True
                    cv.rectangle(img, (x + nx, y + ny), (x + nx + nw, y + ny + nh), color=(0, 0, 255), thickness=2)
                    # 启动线程保存图片
                    if self.is_save:
                        if self.save_thread is None or not self.save_thread.is_alive():
                            self.save_thread = threading.Thread(target=self.save_image,
                                                                args=(detect_area, "./save/mouth", 4))
                            self.save_thread.start()
        else:
            pass

        return img, detect_area, ok

    def draw_nose_rectangle(self, image: cv.Mat):
        # 进行面部特征标记
        face_landmarks_list = face_recognition.face_landmarks(image)

        # 存储鼻子检测框的数组
        nose_rectangles = []
        t = image.copy()

        # 假设只有一张脸
        if len(face_landmarks_list) > 0:
            face_landmarks = face_landmarks_list[0]
            nose_bridge = face_landmarks['nose_bridge']
            nose_tip = face_landmarks['nose_tip']

            # 将坐标转换为整数
            nose_bridge = [(int(x), int(y)) for x, y in nose_bridge]
            nose_tip = [(int(x), int(y)) for x, y in nose_tip]

            # 计算包含所有鼻子特征点的最小矩形
            all_nose_points = np.array(nose_bridge + nose_tip)
            x, y, w, h = cv.boundingRect(all_nose_points)

            # 创建一个空白图像
            image_with_rect = image.copy()

            # 在图像上绘制最小矩形
            cv.rectangle(image_with_rect, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 存储鼻子检测框的数组
            nose_rectangles.append((x, y, w, h))

        else:
            print("No face found in the image.")
            return None, None

        x, y, w, h = nose_rectangles[0]
        roi = t[y:y + h + 5, x - 5:x + w]

        return image_with_rect, roi

    def draw_eyes_rectangle(self, image: cv.Mat):
        # 进行面部特征标记
        face_landmarks_list = face_recognition.face_landmarks(image)

        # 存储眼睛检测框的数组
        eyes_rectangles = []

        t = image.copy()

        # 假设只有一张脸
        if len(face_landmarks_list) > 0:
            face_landmarks = face_landmarks_list[0]

            if 'left_eye' in face_landmarks:
                left_eye = face_landmarks['left_eye']
                left_eye_points = [(int(x), int(y)) for x, y in left_eye]
                x, y, w, h = cv.boundingRect(np.array(left_eye_points))

                # 在图像上绘制眼睛检测框
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 存储眼睛检测框的数组
                eyes_rectangles.append((x, y, w, h))

            if 'right_eye' in face_landmarks:
                right_eye = face_landmarks['right_eye']
                right_eye_points = [(int(x), int(y)) for x, y in right_eye]
                x, y, w, h = cv.boundingRect(np.array(right_eye_points))

                # 在图像上绘制眼睛检测框
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 存储眼睛检测框的数组
                eyes_rectangles.append((x, y, w, h))

        else:
            print("No face found in the image.")
            return None, None

        x, y, w, h = eyes_rectangles[0]
        roi = t[y:y + h + 5, x:x + w + 5]

        return image, roi

    def draw_face_rectangle(self, image: cv.Mat):
        # 进行面部特征标记
        face_landmarks_list = face_recognition.face_landmarks(image)

        # 存储脸部检测框的数组
        face_rectangles = []

        t = image.copy()

        # 假设只有一张脸
        if len(face_landmarks_list) > 0:
            face_landmarks = face_landmarks_list[0]

            # 获取整张脸的关键点
            all_face_points = []
            for facial_feature in face_landmarks.values():
                all_face_points.extend(facial_feature)

            # 将坐标转换为整数
            all_face_points = [(int(x), int(y)) for x, y in all_face_points]

            # 计算包含整张脸的最小矩形
            x, y, w, h = cv.boundingRect(np.array(all_face_points))

            # 创建一个空白图像
            image_with_rect = image.copy()

            # 在图像上绘制最小矩形
            cv.rectangle(image_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 存储脸部检测框的数组
            face_rectangles.append((x, y, w, h))

        else:
            print("No face found in the image.")
            return None, None

        x, y, w, h = face_rectangles[0]
        roi = t[y:y + h, x:x + w]

        return image_with_rect, roi

    def draw_mouth_rectangle(self, image: cv.Mat):
        # 进行面部特征标记
        face_landmarks_list = face_recognition.face_landmarks(image)

        # 存储嘴巴检测框的数组
        mouth_rectangles = []

        t = image.copy()

        # 假设只有一张脸
        if len(face_landmarks_list) > 0:
            face_landmarks = face_landmarks_list[0]

            if 'bottom_lip' in face_landmarks:
                bottom_lip = face_landmarks['bottom_lip']
                bottom_lip_points = [(int(x), int(y)) for x, y in bottom_lip]
                x, y, w, h = cv.boundingRect(np.array(bottom_lip_points))

                # 在图像上绘制嘴巴检测框
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 存储嘴巴检测框的数组
                mouth_rectangles.append((x, y, w, h))

        else:
            print("No face found in the image.")
            return None, None

        x, y, w, h = mouth_rectangles[0]
        roi = t[y - 5:y + h, x + 5:x + w]

        return image, roi
