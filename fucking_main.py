from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QFileDialog, QMainWindow, QButtonGroup
from PyQt5.QtGui import QImage, QPixmap
import sys
import mainwindow
import cv2 as cv
from camera_capture import CameraCapture

class Qt_Window(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.mode = 0
        self.group_radios = QButtonGroup(self)
        self.group_radios.addButton(self.ui.cbx_face)
        self.group_radios.addButton(self.ui.cbx_contours)
        self.group_radios.addButton(self.ui.cbx_eyes)
        self.group_radios.addButton(self.ui.cbx_nose)
        self.group_radios.addButton(self.ui.cbx_mouth)
        self.group_radios.setExclusive(True)
        self.group_radios.buttonClicked.connect(self.on_choice_buttonGroup)

        self.camera_capture = CameraCapture(self.ui.lab_face.width(), self.ui.lab_face.height())
        # 转到操作函数处理
        self.camera_capture.frame_signal.connect(self.ui.lab_face.getMat)
        self.camera_capture.frame_signal_detect.connect(self.ui.lab_dectface.getMat)
        self.ui.lab_dectface.send_detect_area_signal.connect(self.ui.lab_dectarea.getMat)

        self.ui.lab_face.send_image_1.connect(self.save_single_1pixmap)
        self.ui.lab_face.send_image_1.connect(self.ui.lab_dectface.load_image_2)
        self.ui.lab_dectface.send_image_2.connect(self.ui.lab_dectarea.load_image_3)

        # 掩码遮罩
        self.ui.btn_mask.clicked.connect(self.on_btn_mask_clicked)
        self.ui.btn_mask2.clicked.connect(self.on_btn_mask2_clicked)
        self.ui.btn_mask3.clicked.connect(self.on_btn_mask3_clicked)
        self.ui.btn_none_mask.clicked.connect(self.on_btn_none_mask_clicked)

        self.ui.action_open_picture.triggered.connect(self._on_action_open_picture_triggered)
        self.ui.action_open_video.triggered.connect(self._on_action_open_video_triggered)
        self.ui.action_open_camera.triggered.connect(self._on_action_open_camera_triggered)

        # 程序退出
        QApplication.instance().aboutToQuit.connect(self.on_exit)

    def save_single_1pixmap(self, mat):
        self.single_ori_mat = mat.copy()

    def update_display_ori(self, image: QImage):
        self.ui.lab_face.setPixmap(QPixmap.fromImage(image))

    def update_display_detect(self, image: QImage):
        self.ui.lab_dectface.setPixmap(QPixmap.fromImage(image))

    def on_exit(self):
        self.camera_capture.quit()
        QApplication.instance().exit()

    # 菜单行为-------------------------------------------
    def _on_action_open_picture_triggered(self):
        if self.camera_capture.isRunning():
            self.camera_capture.is_running = False
        try:
            fileName, _ = QFileDialog.getOpenFileName(self, "请选择图片资源", ".", "Images (*.png *.xpm *.jpg)")
            if fileName:
                self.mode = 0
                self.single_mt = cv.imread(str(fileName))
                self.ui.lab_face.load_image_1(self.single_mt)
            else:
                pass
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def _on_action_open_video_triggered(self):
        self.mode = 1
        fileName, _ = QFileDialog.getOpenFileName(self, "请选择视频资源", ".", "Videos (*.mp4 *.avi)")
        if fileName:
            if self.camera_capture.cap.isOpened():
                print("打开新的视频")
                self.camera_capture.cap.release()
            self.camera_capture.cap.open(str(fileName))
            self.camera_capture.FPS = self.camera_capture.cap.get(cv.CAP_PROP_FPS)

            self.ui.lab_face.use_video = True
            self.ui.lab_dectface.use_video = True
            self.ui.lab_dectarea.use_video = True
            frame_w = self.camera_capture.cap.get(3)
            frame_h = self.camera_capture.cap.get(4)
            self.ui.lab_face.frame_width = frame_w
            self.ui.lab_face.frame_height = frame_h
            self.ui.lab_dectface.frame_width = frame_w
            self.ui.lab_dectface.frame_height = frame_h
            self.ui.lab_dectarea.frame_width = frame_w
            self.ui.lab_dectarea.frame_height = frame_h

            self.camera_capture.start()
            self.camera_capture.is_running = True

    def _on_action_open_camera_triggered(self):
        # 摄像头开始
        self.mode = 2
        self.camera_capture.is_running = True
        self.camera_capture.start()

    def on_action_exit_triggered(self):
        self.on_exit()

    # end 菜单行为-------------------------------------------

    # 脸部遮罩-------------------------------------------
    def on_btn_mask_clicked(self):
        self.ui.lab_dectface.use_mask = True
        self.ui.lab_dectface.mask_image = cv.imread('assets/mask1.png')

    def on_btn_mask2_clicked(self):
        self.ui.lab_dectface.mask_image = cv.imread('assets/mask2.jpg')

    def on_btn_mask3_clicked(self):
        self.ui.lab_dectface.mask_image = cv.imread('assets/mask3.png')

    def on_btn_none_mask_clicked(self):
        self.ui.lab_dectface.use_mask = False

    def on_choice_buttonGroup(self, button):
        if button == self.ui.cbx_face:
            self.ui.lab_dectface.cbx_id = 0
        elif button == self.ui.cbx_contours:
            self.ui.lab_dectface.cbx_id = 1
        elif button == self.ui.cbx_eyes:
            self.ui.lab_dectface.cbx_id = 2
        elif button == self.ui.cbx_nose:
            self.ui.lab_dectface.cbx_id = 3
        elif button == self.ui.cbx_mouth:
            self.ui.lab_dectface.cbx_id = 4

        if self.mode == 0:
            self.ui.lab_dectface.load_image_2(self.single_ori_mat)

    def on_btn_save_released(self):
        self.ui.lab_dectface.is_save = not self.ui.lab_dectface.is_save


if __name__ == '__main__':
    myApp = QApplication(sys.argv)
    myWindow = Qt_Window()
    myWindow.show()
    sys.exit(myApp.exec_())
