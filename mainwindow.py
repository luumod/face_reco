# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1502, 785)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.tab)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.widget = QtWidgets.QWidget(self.tab)
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_3 = QtWidgets.QWidget(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.widget_3)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.lab_face = VideoWidget(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lab_face.sizePolicy().hasHeightForWidth())
        self.lab_face.setSizePolicy(sizePolicy)
        self.lab_face.setMinimumSize(QtCore.QSize(480, 480))
        self.lab_face.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.lab_face.setStyleSheet("QLabel {\n"
"    border: 2px solid black;\n"
"    padding: 2px; /* 如果需要增加内边距，可选 */\n"
"}\n"
"")
        self.lab_face.setText("")
        self.lab_face.setObjectName("lab_face")
        self.verticalLayout_2.addWidget(self.lab_face)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 9)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.widget_3)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.lab_dectface = VideoWidget(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lab_dectface.sizePolicy().hasHeightForWidth())
        self.lab_dectface.setSizePolicy(sizePolicy)
        self.lab_dectface.setMinimumSize(QtCore.QSize(480, 480))
        self.lab_dectface.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.lab_dectface.setStyleSheet("QLabel {\n"
"    border: 2px solid black;\n"
"    padding: 2px; /* 如果需要增加内边距，可选 */\n"
"}\n"
"")
        self.lab_dectface.setText("")
        self.lab_dectface.setObjectName("lab_dectface")
        self.verticalLayout_3.addWidget(self.lab_dectface)
        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 9)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_3 = QtWidgets.QLabel(self.widget_3)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.lab_dectarea = VideoWidget(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lab_dectarea.sizePolicy().hasHeightForWidth())
        self.lab_dectarea.setSizePolicy(sizePolicy)
        self.lab_dectarea.setMinimumSize(QtCore.QSize(200, 200))
        self.lab_dectarea.setStyleSheet("QLabel {\n"
"    border: 2px solid black;\n"
"    padding: 2px; /* 如果需要增加内边距，可选 */\n"
"}\n"
"")
        self.lab_dectarea.setText("")
        self.lab_dectarea.setAlignment(QtCore.Qt.AlignCenter)
        self.lab_dectarea.setObjectName("lab_dectarea")
        self.verticalLayout_4.addWidget(self.lab_dectarea)
        self.label_4 = QtWidgets.QLabel(self.widget_3)
        self.label_4.setText("")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_4.addWidget(self.label_4)
        self.verticalLayout_4.setStretch(0, 1)
        self.verticalLayout_4.setStretch(1, 5)
        self.verticalLayout_4.setStretch(2, 1)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        self.verticalLayout.addWidget(self.widget_3)
        self.widget_2 = QtWidgets.QWidget(self.widget)
        self.widget_2.setMinimumSize(QtCore.QSize(150, 150))
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.groupBox_2 = QtWidgets.QGroupBox(self.widget_2)
        self.groupBox_2.setObjectName("groupBox_2")
        self.btn_mask = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_mask.setGeometry(QtCore.QRect(40, 40, 75, 24))
        self.btn_mask.setObjectName("btn_mask")
        self.btn_mask2 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_mask2.setGeometry(QtCore.QRect(130, 40, 75, 24))
        self.btn_mask2.setObjectName("btn_mask2")
        self.btn_mask3 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_mask3.setGeometry(QtCore.QRect(220, 40, 75, 24))
        self.btn_mask3.setObjectName("btn_mask3")
        self.btn_none_mask = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_none_mask.setGeometry(QtCore.QRect(310, 40, 75, 24))
        self.btn_none_mask.setObjectName("btn_none_mask")
        self.btn_save = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_save.setGeometry(QtCore.QRect(300, 80, 131, 24))
        self.btn_save.setObjectName("btn_save")
        self.horizontalLayout_5.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.widget_2)
        self.groupBox_3.setObjectName("groupBox_3")
        self.cbx_eyes = QtWidgets.QRadioButton(self.groupBox_3)
        self.cbx_eyes.setGeometry(QtCore.QRect(110, 20, 91, 16))
        self.cbx_eyes.setObjectName("cbx_eyes")
        self.cbx_face = QtWidgets.QRadioButton(self.groupBox_3)
        self.cbx_face.setGeometry(QtCore.QRect(10, 25, 61, 16))
        self.cbx_face.setChecked(True)
        self.cbx_face.setObjectName("cbx_face")
        self.cbx_mouth = QtWidgets.QRadioButton(self.groupBox_3)
        self.cbx_mouth.setGeometry(QtCore.QRect(10, 100, 71, 16))
        self.cbx_mouth.setObjectName("cbx_mouth")
        self.cbx_contours = QtWidgets.QRadioButton(self.groupBox_3)
        self.cbx_contours.setGeometry(QtCore.QRect(10, 60, 61, 16))
        self.cbx_contours.setObjectName("cbx_contours")
        self.cbx_nose = QtWidgets.QRadioButton(self.groupBox_3)
        self.cbx_nose.setGeometry(QtCore.QRect(110, 60, 61, 16))
        self.cbx_nose.setObjectName("cbx_nose")
        self.horizontalLayout_5.addWidget(self.groupBox_3)
        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 1)
        self.verticalLayout.addWidget(self.widget_2)
        self.verticalLayout.setStretch(1, 1)
        self.horizontalLayout_4.addWidget(self.widget)
        self.horizontalLayout_4.setStretch(0, 4)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1502, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_open_picture = QtWidgets.QAction(MainWindow)
        self.action_open_picture.setObjectName("action_open_picture")
        self.action_open_video = QtWidgets.QAction(MainWindow)
        self.action_open_video.setObjectName("action_open_video")
        self.action_open_camera = QtWidgets.QAction(MainWindow)
        self.action_open_camera.setObjectName("action_open_camera")
        self.action_exit = QtWidgets.QAction(MainWindow)
        self.action_exit.setObjectName("action_exit")
        self.menu.addAction(self.action_open_picture)
        self.menu.addAction(self.action_open_video)
        self.menu.addAction(self.action_open_camera)
        self.menu.addSeparator()
        self.menu.addAction(self.action_exit)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "原图片"))
        self.label_2.setText(_translate("MainWindow", "目标检测"))
        self.label_3.setText(_translate("MainWindow", "检测结果"))
        self.groupBox_2.setTitle(_translate("MainWindow", "脸部遮罩（摄像头）"))
        self.btn_mask.setText(_translate("MainWindow", "图片1"))
        self.btn_mask2.setText(_translate("MainWindow", "图片2"))
        self.btn_mask3.setText(_translate("MainWindow", "图片3"))
        self.btn_none_mask.setText(_translate("MainWindow", "无遮罩"))
        self.btn_save.setText(_translate("MainWindow", "保存检测图片"))
        self.groupBox_3.setTitle(_translate("MainWindow", "部位检测"))
        self.cbx_eyes.setText(_translate("MainWindow", "眼睛"))
        self.cbx_face.setText(_translate("MainWindow", "脸部"))
        self.cbx_mouth.setText(_translate("MainWindow", "嘴巴"))
        self.cbx_contours.setText(_translate("MainWindow", "轮廓"))
        self.cbx_nose.setText(_translate("MainWindow", "鼻子"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Tab 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Tab 2"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.action_open_picture.setText(_translate("MainWindow", "打开图片"))
        self.action_open_video.setText(_translate("MainWindow", "打开视频"))
        self.action_open_camera.setText(_translate("MainWindow", "打开摄像头"))
        self.action_exit.setText(_translate("MainWindow", "退出"))
from video_widget import VideoWidget
