# -*- coding: utf-8 -*-
'''
人脸&表情 识别主窗口
Author:我是一条咸鱼
Date:2020-09-02
Mail:AndrewWC666@gmail.com
'''
import sys
import time

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ui_mainwindow import Ui_mainWindow
import cv2
import numpy as np
import tensorflow as tf
from my_utils import *



class MainWindow(QMainWindow,Ui_mainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.scene = QGraphicsScene()
        self.video_haar_detector = VideoHaarDetector()
        self.image_haar_detector = ImageHaarDetector()
        self.lbph_trainer = LbphTrainer()
        self.image_lbph_recognizer = ImageLbphRecognizer()
        self.video_lbph_recognizer = VideoLbphRecognizer()
        self.load_mtcnn = LoadMtcnn()
        self.image_mtcnn_detector = ImageMtcnnDetector()
        self.video_mtcnn_detector = VideoMtcnnDetector()
        self.load_facenet = LoadFaceNet()
        self.image_facenet_recognizer = ImageFaceNetRecognizer()
        self.video_facenet_recognizer = VideoFaceNetRecognizer()
        # self.load_keras = LoadKeras()
        self.image_emotion_recognizer = ImageEmotionRecognizer()
        self.video_emotion_recognizer = VideoEmotionRecognizer()

        self.img = None
        self.recognizer = None              #opencv识别器
        self.name_dict = None
        self.detector = None                #mtcnn检测器
        self.facenet_recognizer = None
        self.dataset_emb = None
        self.names_list = None
        # self.fer_model = None
        # self.graph = tf.get_default_graph()



        #信号与槽的绑定
        self.pushButton.clicked.connect(self.onClicked_button)
        self.pushButton_2.clicked.connect(self.onClicked_button_2)
        self.radioButton.toggled.connect(self.radio_state_change)
        self.video_haar_detector.update_frame.connect(self.update_graphics_view)
        self.image_haar_detector.update_image.connect(self.update_graphics_view)
        self.lbph_trainer.update_image.connect(self.get_detector_recognizer)
        self.image_lbph_recognizer.update_image.connect(self.update_graphics_view)
        self.image_lbph_recognizer.update_image.connect(self.update_table_view)
        self.video_lbph_recognizer.update_frame.connect(self.update_graphics_view)
        self.video_lbph_recognizer.update_frame.connect(self.update_table_view)
        self.load_mtcnn.load_finished.connect(self.get_detector_recognizer)
        self.image_mtcnn_detector.update_image.connect(self.update_graphics_view)
        self.video_mtcnn_detector.update_frame.connect(self.update_graphics_view)
        self.load_facenet.load_finished.connect(self.get_facenet_recognizer)
        self.image_facenet_recognizer.update_image.connect(self.update_graphics_view)
        self.image_facenet_recognizer.update_image.connect(self.update_table_view)
        self.video_facenet_recognizer.update_frame.connect(self.update_graphics_view)
        self.video_facenet_recognizer.update_frame.connect(self.update_table_view)
        # self.load_keras.load_finished.connect(self.get_fer_model)
        self.image_emotion_recognizer.update_image.connect(self.update_graphics_view)
        self.image_emotion_recognizer.update_image.connect(self.update_table_view)
        self.video_emotion_recognizer.update_frame.connect(self.update_graphics_view)
        self.video_emotion_recognizer.update_frame.connect(self.update_table_view)

        self.lbph_trainer.start()
        self.load_mtcnn.start()
        self.load_facenet.start()
        # self.load_keras.start()
        print('开始训练')

        self.setWindowIcon(QIcon('face.jpg'))
        self.show()

    def onClicked_button(self):
        '''
        函数功能:"打开文件"按钮的槽函数,打开文件对话框
        参数:实例对象
        返回值:无
        '''
        try:
            goal_type = '图像文件(*.jpg *.png)'
            fname,is_cancel = QFileDialog.getOpenFileName(self,'打开文件','.',goal_type)
            self.tableWidget.clear()

            if is_cancel == goal_type:     #如果未选择图片，关闭文件对话框，则不输出任何信息
                show_info = show_time() + '  #OK  IMAGE_PATH:  ' + fname
                self.plainTextEdit.appendPlainText(show_info)

                code = get_current_radio_combobox_state(self)
                if 0 == code:
                    self.image_haar_detector.image_path = fname
                    self.image_haar_detector.start()
                elif 1 == code:
                    if self.detector == None:
                        self.plainTextEdit.appendPlainText(show_time() + '  #ERROR  模型未加载完毕!')
                    else:
                        self.image_mtcnn_detector.image_path = fname
                        self.image_mtcnn_detector.detector = self.detector
                        self.image_mtcnn_detector.start()
                elif 2 == code:
                    if self.recognizer == None:
                        self.plainTextEdit.appendPlainText(show_time() + '  #ERROR  模型未训练完毕!')
                    else:
                        self.image_lbph_recognizer.image_path = fname
                        self.image_lbph_recognizer.recognizer = self.recognizer
                        self.image_lbph_recognizer.start()
                elif 3 == code:
                    if self.facenet_recognizer is not None:
                        self.image_facenet_recognizer.recognizer = self.facenet_recognizer
                        self.image_facenet_recognizer.image_path = fname
                        self.image_facenet_recognizer.detector = self.detector
                        self.image_facenet_recognizer.dataset_emb = self.dataset_emb
                        self.image_facenet_recognizer.names_list = self.names_list
                        self.image_facenet_recognizer.start()
                    else:
                        self.plainTextEdit.appendPlainText(show_time() + '  #ERROR  模型未加载完毕!')
                elif 4 == code:
                    # if self.fer_model is not None:
                    self.image_emotion_recognizer.recognizer = self.facenet_recognizer
                    self.image_emotion_recognizer.detector = self.detector
                    self.image_emotion_recognizer.image_path = fname
                    self.image_emotion_recognizer.dataset_emb = self.dataset_emb
                    self.image_emotion_recognizer.names_list = self.names_list
                    # self.image_emotion_recognizer.fer_model = self.fer_model
                    self.image_emotion_recognizer.start()
                else:
                    self.plainTextEdit.appendPlainText(show_time() + '  #ERROR  操作执行失败!')
        except:
           self.plainTextEdit.appendPlainText(show_time() + '  #ERROR  图片打开失败!')

    def onClicked_button_2(self):
        '''
        函数功能:"打开摄像头"按钮的槽函数,打开摄像头,获取图像
        参数:实例对象
        返回值:无
        '''
        if self.pushButton_2.text() == '打开摄像头':
            code = get_current_radio_combobox_state(self)
            if 0 == code:
                try:
                    self.video_haar_detector.capture = True
                    self.video_haar_detector.start()
                    self.pushButton_2.setText('关闭摄像头')
                    self.plainTextEdit.appendPlainText(show_time() + '  #OK 摄像头已打开.')
                except:
                    self.plainTextEdit.appendPlainText(show_time() + '  #ERROR 摄像头打开失败!')

            elif 1 == code:
                if self.detector == None:
                    print('未加载完毕')
                else:
                    self.video_mtcnn_detector.capture = True
                    self.video_mtcnn_detector.detector = self.detector
                    self.video_mtcnn_detector.start()
                    self.pushButton_2.setText('关闭摄像头')
            elif 2 == code:
                if self.recognizer == None:
                    print('未训练完毕!!')
                else:
                    self.video_lbph_recognizer.capture = True
                    self.video_lbph_recognizer.recognizer = self.recognizer
                    self.video_lbph_recognizer.start()
                    self.pushButton_2.setText('关闭摄像头')
            elif 3 == code:
                if self.facenet_recognizer is not None:
                    self.video_facenet_recognizer.capture = True
                    self.video_facenet_recognizer.recognizer = self.facenet_recognizer
                    self.video_facenet_recognizer.detector = self.detector
                    self.video_facenet_recognizer.dataset_emb = self.dataset_emb
                    self.video_facenet_recognizer.names_list = self.names_list
                    self.video_facenet_recognizer.start()
                    self.pushButton_2.setText('关闭摄像头')
                else:
                    print('未加载完毕!')
            elif 4 == code:
                self.video_emotion_recognizer.capture = True
                self.video_emotion_recognizer.recognizer = self.facenet_recognizer
                self.video_emotion_recognizer.detector = self.detector
                self.video_emotion_recognizer.dataset_emb = self.dataset_emb
                self.video_emotion_recognizer.names_list = self.names_list
                self.video_emotion_recognizer.start()
                self.pushButton_2.setText('关闭摄像头')

            else:
                self.plainTextEdit.appendPlainText(show_time() + '  #ERROR  操作执行失败!')
        else:
            self.pushButton_2.setText('打开摄像头')
            self.tableWidget.clearContents()
            self.video_haar_detector.capture = False
            self.video_lbph_recognizer.capture = False
            self.video_mtcnn_detector.capture = False
            self.video_facenet_recognizer.capture = False
            self.video_emotion_recognizer.capture = False
            self.plainTextEdit.appendPlainText(show_time() + '  #OK 摄像头已关闭.')

    def update_graphics_view(self,signal_info):
        '''
        函数功能:子线程槽函数,接收子线程发送的图像,更新UI
        参数:numpy图片
        返回值:无
        '''
        img = signal_info.get('image')   # img 是一个numpy类型的图片

        if img is not None:
            self.img = QImage(img,img.shape[1],img.shape[0],QImage.Format_BGR888)
            img = self.img.scaled(self.graphicsView.width(),self.graphicsView.height(),Qt.IgnoreAspectRatio,Qt.SmoothTransformation)
            pix = QPixmap.fromImage((img))
            self.item = QGraphicsPixmapItem(pix)
            self.scene.addItem((self.item))

            self.graphicsView.setScene(self.scene)
            self.graphicsView.update()

            if signal_info.get('detected_face_num') is not None:
                self.plainTextEdit.appendPlainText(show_time() + '  #OK 共检测到' + str(signal_info.get('detected_face_num')) + '张人脸. 用时' + \
                                                    str(signal_info.get('time_cost')) + 's.' )


        else:
                self.plainTextEdit.appendPlainText(show_time() + '  #OK 未检测到人脸. ')
    def update_table_view(self,signal_info):
        #识别子线程,所用
        results = signal_info.get('results')
        if results is not None:
            name = []
            for result in results:
                if result[1] > 60.0:
                    name.append(self.name_dict.get(result[0]))
                else:
                    name.append('Unknow')

            self.tableWidget.setRowCount(len(name))
            for i in range(len(name)):
                self.tableWidget.setItem(i,0,QTableWidgetItem(name[i]))

        if signal_info.get('pred_names') is not None:
            names = signal_info.get('pred_names')
            self.tableWidget.setRowCount(len(names))
            for i,name in enumerate(names):
                self.tableWidget.setItem(i,0,QTableWidgetItem(name))

        if signal_info.get('emotions') is not None:
            emotions = signal_info.get('emotions')
            for i,emotion in enumerate(emotions):
                self.tableWidget.setItem(i,1,QTableWidgetItem(emotion))
    def get_fer_model(self,signal_info):
        self.fer_model = signal_info.get('fer_model')
        time_cost = signal_info.get('time_cost')
        self.plainTextEdit.appendPlainText(show_time() + '  #OK 表情识别模型加载完毕 用时' + str(time_cost) + 's.')

    def get_detector_recognizer(self,signal_info):
        time_cost = signal_info.get('time_cost')
        if signal_info.get('detector') is not None:
            self.detector = signal_info.get('detector')
            self.plainTextEdit.appendPlainText(show_time() + '  #OK MTCNN加载完毕 用时' + str(time_cost) + 's.')
        else:
            self.recognizer = signal_info.get('recognizer')
            self.plainTextEdit.appendPlainText(show_time() + '  #OK LBPH训练完毕 用时' + str(time_cost) + 's.')

        if signal_info.get('name_dict') is not None:
            self.name_dict = signal_info.get('name_dict')

    def get_facenet_recognizer(self,signal_info):
        self.facenet_recognizer = signal_info.get('recognizer')
        self.dataset_emb = signal_info.get('dataset_emb')
        self.names_list = signal_info.get('names_list')
        time_cost = signal_info.get('time_cost')
        self.plainTextEdit.appendPlainText(show_time() + '  #OK FaceNet网络加载完毕 用时' + str(time_cost) + 's.')

    def radio_state_change(self):
        '''
        函数功能:radioButton槽函数,更新comboBox信息
        参数:实例对象
        返回值:无
        '''
        isChecked = self.radioButton.isChecked()
        if isChecked == True:
            self.comboBox.clear()
            self.tableWidget.clearContents()
            self.comboBox.addItems(['Haar检测器','MTCNN'])
        else:
            self.comboBox.clear()
            self.comboBox.addItems(['LBPH识别','FaceNet','表情识别'])

    def resizeEvent(self,event):
        '''
        函数功能：调整QGraphicsView内图片的大小，由系统调用
        参数：实例对象，当前窗口大小
        返回值：无
        '''
        # print(self.graphicsView.width(),self.graphicsView.height())
        if self.img != None:
            img = self.img.scaled(self.graphicsView.width(),self.graphicsView.height(),Qt.IgnoreAspectRatio,Qt.SmoothTransformation)
            pix = QPixmap.fromImage((img))
            self.item = QGraphicsPixmapItem(pix)
            self.scene.addItem((self.item))
            self.graphicsView.setScene(self.scene)
            self.graphicsView.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    sys.exit(app.exec_())
