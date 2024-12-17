from PySide6 import QtWidgets, QtCore, QtGui
import cv2 as cv
import os, time
from threading import Thread
from ultralytics import YOLO

os.environ['YOLO_VERBOSE'] = 'False'


class MWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # 设置UI界面
        self.setupUI()
        # 将摄像头，视频文件，停止按钮与（startCamera）（startVideoFile）（stop）三个函数关联成点击时触发
        self.camBtn.clicked.connect(self.startCamera)
        self.videoBtn.clicked.connect(self.startVideoFile)
        self.stopBtn.clicked.connect(self.stop)
        self.picture.clicked.connect(self.startPicture)  # 连接“照片”按钮

        self.timer_camera = QtCore.QTimer()  # 定时器，用来控制显示摄像头视频的帧率
        self.timer_camera.timeout.connect(self.show_camera)

        self.model = YOLO(
          "D:\yolo\yolo8/ultralytics-main/ultralytics-main/anan_domo/yolov8n.pt")  # 加载 YOLO 模型

        self.frameToAnalyze = []  # 存放待处理的视频帧
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()  # 启动处理视频帧的线程

        self.timer_videoFile = QtCore.QTimer()  # 定时器，用来控制视频文件的显示帧率
        self.timer_videoFile.timeout.connect(self.show_videoFile)

        self.vframeIdx = 0  # 当前要播放的视频帧号
        self.cap = None  # cv.VideoCapture 实例
        self.stopFlag = False  # 停止标志

    def setupUI(self):
        self.resize(1200, 800)  # 设置应用窗口的初始大小
        self.setWindowTitle('YOLO_anan')  # 标题栏

        centralWidget = QtWidgets.QWidget(self)  # 创建中央控件
        self.setCentralWidget(centralWidget)

        mainLayout = QtWidgets.QVBoxLayout(centralWidget)  # 创建垂直布局

        topLayout = QtWidgets.QHBoxLayout()  # 创建水平布局
        self.label_ori_video = QtWidgets.QLabel(self)  # 创建标签用于显示原始视频
        self.label_treated = QtWidgets.QLabel(self)  # 创建标签用于显示处理后的图像
        self.label_ori_video.setFixedSize(520, 400)
        self.label_treated.setFixedSize(520, 400)
        self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
        self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')

        topLayout.addWidget(self.label_ori_video)
        topLayout.addWidget(self.label_treated)

        mainLayout.addLayout(topLayout)

        groupBox = QtWidgets.QGroupBox(self)  # 创建QGroupBox容器控件
        bottomLayout = QtWidgets.QHBoxLayout(groupBox)  # 创建水平布局
        self.textLog = QtWidgets.QTextBrowser()  # 创建一个文本浏览器控件
        bottomLayout.addWidget(self.textLog)

        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QVBoxLayout()  # 创建垂直布局用于按钮
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn = QtWidgets.QPushButton('📹摄像头')
        self.stopBtn = QtWidgets.QPushButton('🛑停止')
        self.picture = QtWidgets.QPushButton("选择照片")
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.stopBtn)
        btnLayout.addWidget(self.picture)  # 添加选择照片按钮
        bottomLayout.addLayout(btnLayout)

    def startCamera(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            print("摄像头不能打开")
            return

        if not self.timer_camera.isActive():
            self.timer_camera.start(30)
            self.stopFlag = False

    def show_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv.resize(frame, (520, 400))
        self.setFrameToOriLabel(frame)

    def setFrameToOriLabel(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # 转换颜色格式
        qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                              QtGui.QImage.Format_RGB888)
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage))

        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame)

    def frameAnalyzeThreadFunc(self):
        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue

            frame = self.frameToAnalyze.pop(0)
            results = self.model(frame)[0]  # 使用YOLO进行目标检测
            img = results.plot(line_width=1)  # 绘制检测框
            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                  QtGui.QImage.Format_RGB888)

            if not self.stopFlag:
                self.label_treated.setPixmap(QtGui.QPixmap.fromImage(qImage))

            time.sleep(0.5)

    def stop(self):
        self.stopFlag = True
        self.timer_camera.stop()
        self.timer_videoFile.stop()

        if self.cap:
            self.cap.release()
        self.label_ori_video.clear()
        self.label_treated.clear()

    def startVideoFile(self):
        self.stop()
        videoPath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频文件", ".", "视频文件 (*.mp4 *.avi)")
        if not videoPath:
            return

        self.cap = cv.VideoCapture(videoPath)
        if not self.cap.isOpened():
            print("打开视频文件失败")
            return

        self.timer_videoFile.start(30)
        self.stopFlag = False

    def startPicture(self):
        self.stop()
        picturePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", ".",
                                                               "图片文件 (*.png *.jpg *.jpeg *.bmp)")
        if not picturePath:
            return

        image = cv.imread(picturePath)  # 读取图片
        if image is None:
            print("读取图片失败")
            return
        height, width, _ = image.shape

        self.label_ori_video.setFixedSize(width, height)#原图片区域
        self.label_treated.setFixedSize(width, height)#标记后的图片区域

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.model(image_rgb)[0]  # 使用YOLO进行目标检测
        img = results.plot(line_width=1)  # 绘制检测框

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # YOLO处理后的图像
        qImage = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0],
                              QtGui.QImage.Format_RGB888)

        self.label_treated.setPixmap(QtGui.QPixmap.fromImage(qImage))

    def show_videoFile(self):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.vframeIdx)  # 跳到指定帧
        self.vframeIdx += 1
        ret, frame = self.cap.read()  # 读取当前帧

        if not ret:
            self.stop()
            return

        self.setFrameToOriLabel(frame)


app = QtWidgets.QApplication([])
window = MWindow()
window.show()
app.exec()
