# -*- coding: utf-8 -*-
'''
Author:我是一条咸鱼
Date:2020-09-02
'''

import time
import os

from PyQt5.QtCore import QThread,pyqtSignal
import cv2
import numpy as np
import keras
from keras.preprocessing import image
import tensorflow as tf

from face_recognition import Facedetection,FacenetEmbedding
from utils import image_processing,file_processing

def show_time():
    '''
    函数功能:返回当前时间(时:分:秒)
    参数:无
    返回值:格式化的时间
    '''
    cur_time = time.strftime("%H:%M:%S",time.localtime())
    return '[' + cur_time + ']'

############################################################# 子线程类 START ##########################################################

#自定义子线程信号,处理摄像头捕获的图像
class VideoHaarDetector(QThread):
    update_frame = pyqtSignal(dict)  #定义一个信号,触发时传入一个字典

    capture = True
    def run(self):
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        while self.capture:
            start = time.time()
            ret,frame = cap.read()
            if not ret:
                self.update_frame.emit('ERROR')
                break

            frame,faces = opencv_face_detector(frame)
            end = time.time()
            time_cost = round((end - start),2)
            detected_face_num = len(faces)
            self.update_frame.emit({'image':frame,'detected_face_num':detected_face_num,'time_cost':time_cost})

        #通过标志位 关闭摄像头会有3s的延迟
        self.update_frame.emit({'image':cv2.imread('./sources/Desert.jpg')})

class ImageHaarDetector(QThread):
    def __init__(self):
        super(ImageHaarDetector,self).__init__()
        self.image_path = None

    update_image = pyqtSignal(dict)

    def run(self):
        start = time.time()
        image = cv2.imread(self.image_path)
        img,faces = opencv_face_detector(image)
        detected_face_num = len(faces)
        end = time.time()
        time_cost = round((end - start),2)
        self.update_image.emit({'image':img,'detected_face_num':detected_face_num,'time_cost':time_cost})

class LbphTrainer(QThread):
    update_image = pyqtSignal(dict)

    def run(self):
        start = time.time()
        train_images,train_labels,name_dict = get_train_data()

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(train_images,train_labels)
        end = time.time()
        time_cost = round((end - start),2)

        self.update_image.emit({'time_cost':time_cost,'recognizer':recognizer,'name_dict':name_dict})

class ImageLbphRecognizer(QThread):
    def __init__(self):
        super(ImageLbphRecognizer,self).__init__()
        self.recognizer = None
        self.image_path = None

    update_image = pyqtSignal(dict)
    def run(self):
        start = time.time()
        image = cv2.imread(self.image_path)
        img,faces = opencv_face_detector(image)
        detected_face_num = len(faces)
        results = opencv_face_recognizer(img,faces,self.recognizer)
        end = time.time()
        time_cost = round((end - start),2)
        self.update_image.emit({'image':img,'detected_face_num':detected_face_num,'time_cost':time_cost,'results':results})

class VideoLbphRecognizer(QThread):
    def __init__(self):
        super(VideoLbphRecognizer,self).__init__()
        self.recognizer = None
        self.image_path = None

    update_frame = pyqtSignal(dict)

    capture = True
    def run(self):
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        while self.capture:
            start = time.time()
            ret,frame = cap.read()
            if not ret:
                self.update_frame.emit('ERROR')
                break

            frame,faces = opencv_face_detector(frame)
            detected_face_num = len(faces)
            results = opencv_face_recognizer(frame,faces,self.recognizer)
            end = time.time()
            time_cost = round((end - start),2)
            self.update_frame.emit({'image':frame,'detected_face_num':detected_face_num,'time_cost':time_cost,'results':results})

        self.update_frame.emit({'image':cv2.imread('./sources/Desert.jpg')})

class LoadMtcnn(QThread):
    load_finished = pyqtSignal(dict)

    def run(self):
        start = time.time()
        face_detector = Facedetection()
        end = time.time()
        time_cost = round((end - start),2)
        self.load_finished.emit({'detector':face_detector,'time_cost':time_cost})

class ImageMtcnnDetector(QThread):
    def __init__(self):
        super(ImageMtcnnDetector,self).__init__()
        self.image_path = None
        self.detector = None

    update_image = pyqtSignal(dict)

    def run(self):
        start = time.time()
        img = image_processing.read_image(self.image_path)
        bboxes,landmarks_list = self.detector.detect_face(img)
        detected_face_num = len(bboxes)

        for (x1,y1,x2,y2) in bboxes:
            img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        end = time.time()
        time_cost = round((end - start),2)
        self.update_image.emit({'image':img,'detected_face_num':detected_face_num,'time_cost':time_cost})

class VideoMtcnnDetector(QThread):
    def __init__(self):
        super(VideoMtcnnDetector,self).__init__()
        self.image_path = None
        self.detector = None

    update_frame = pyqtSignal(dict)

    capture = True
    def run(self):
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        while self.capture:
            start = time.time()
            ret,frame = cap.read()
            if not ret:
                self.update_frame.emit('ERROR')
                break
            img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            bboxes,landmarks_list = self.detector.detect_face(img)
            detected_face_num = len(bboxes)

            for (x1,y1,x2,y2) in bboxes:
                img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            end = time.time()
            time_cost = round((end - start),2)
            self.update_frame.emit({'image':img,'detected_face_num':detected_face_num,'time_cost':time_cost})


        #通过标志位 关闭摄像头会有3s的延迟
        self.update_frame.emit({'image':cv2.imread('./sources/Desert.jpg')})

class LoadFaceNet(QThread):
    def __init__(self):
        super(LoadFaceNet,self).__init__()
        self.model_path = 'models/20180402-114759'
        self.dataset_path = 'dataset/emb/faceEmbedding.npy'
        self.filename = 'dataset/emb/name.txt'

    load_finished = pyqtSignal(dict)
    def run(self):
        start = time.time()
        # #加载数据库的数据
        dataset_emb, names_list = load_dataset(self.dataset_path, self.filename)
        # 初始化facenet
        face_net = FacenetEmbedding(self.model_path)

        end = time.time()
        time_cost = round((end - start),2)
        self.load_finished.emit({'recognizer':face_net,'dataset_emb':dataset_emb,'names_list':names_list,'time_cost':time_cost})

class ImageFaceNetRecognizer(QThread):
    def __init__(self):
        super(ImageFaceNetRecognizer,self).__init__()
        self.recognizer = None
        self.image_path = None
        self.detector = None
        self.dataset_emb = None
        self.names_list = None

    update_image = pyqtSignal(dict)

    def run(self):
        start = time.time()
        image = image_processing.read_image(self.image_path)
        bboxes,pred_names = recognition(image,self.dataset_emb,self.names_list,self.detector,self.recognizer)
        end = time.time()
        time_cost = round((end - start),2)

        for index,(x1,y1,x2,y2) in enumerate(bboxes):
            image = cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(image,'num:' + str(index + 1),(x1,y1-10),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),4)

        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        self.update_image.emit({'image':image,'pred_names':pred_names,'time_cost':time_cost})

class VideoFaceNetRecognizer(QThread):
    def __init__(self):
        super(VideoFaceNetRecognizer,self).__init__()
        self.recognizer = None
        self.detector = None
        self.dataset_emb = None
        self.names_list = None
        self.bboxes = None
        self.pred_names = None
        self.detected_face_num = None

    update_frame = pyqtSignal(dict)

    capture = True
    def run(self):
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        while self.capture:
            start = time.time()
            ret,frame = cap.read()
            if not ret:
                self.update_frame.emit('ERROR')
                break

            results = recognition(frame,self.dataset_emb,self.names_list,self.detector,self.recognizer)
            if results is not None:
                self.bboxes = results[0]
                self.pred_names = results[1]
                for index,(x1,y1,x2,y2) in enumerate(self.bboxes):
                    frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,'num:' + str(index + 1),(x1,y1-10),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),4)

            end = time.time()
            time_cost = round((end - start),2)
            if self.bboxes is not None:
                self.detected_face_num = len(self.bboxes)
                self.update_frame.emit({'image':frame,'pred_names':self.pred_names,'detected_face_num':self.detected_face_num,'time_cost':time_cost})

        #通过标志位 关闭摄像头会有3s的延迟
        self.update_frame.emit({'image':cv2.imread('./sources/Desert.jpg')})

class ImageEmotionRecognizer(QThread):
    def __init__(self):
        super(ImageEmotionRecognizer,self).__init__()
        self.graph = tf.get_default_graph()
        self.recognizer = None
        self.image_path = None
        self.detector = None
        self.dataset_emb = None
        self.names_list = None
        self.fer_model = load_model('./my_weights.h5')

    update_image = pyqtSignal(dict)
    def run(self):
        start = time.time()
        image = image_processing.read_image(self.image_path)
        results = fer_recognition(image,self.dataset_emb,self.names_list,self.detector,self.recognizer)
        pred_names,emotions,image= get_names_and_emotions(image,results,self.fer_model,self.graph)
        end = time.time()
        time_cost = round((end - start),2)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        self.update_image.emit({'image':image,'pred_names':pred_names,'time_cost':time_cost,'emotions':emotions})

class VideoEmotionRecognizer(QThread):
    def __init__(self):
        super(VideoEmotionRecognizer,self).__init__()
        self.graph = tf.get_default_graph()
        self.recognizer = None
        self.detector = None
        self.dataset_emb = None
        self.names_list = None
        self.fer_model = load_model('./my_weights.h5')

    update_frame = pyqtSignal(dict)
    capture = True
    def run(self):
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        while self.capture:
            start = time.time()
            ret,frame = cap.read()
            results = fer_recognition(frame,self.dataset_emb,self.names_list,self.detector,self.recognizer)
            re_value = get_names_and_emotions(frame,results,self.fer_model,self.graph)
            if -1 != re_value:
                pred_names,emotions,frame= re_value[0],re_value[1],re_value[2]
                end = time.time()
                time_cost = round((end - start),2)
                self.update_frame.emit({'image':frame,'pred_names':pred_names,'time_cost':time_cost,'emotions':emotions})
            else:
                self.update_frame.emit({'image':frame})
        self.update_frame.emit({'image':cv2.imread('./sources/Desert.jpg')})

############################################################# 子线程类 END ##########################################################
def get_names_and_emotions(image,results,fer_model,graph):
    EMOTION = {'0':'angry','1':'disgust','2':'fear','3':'happy','4':'sad','5':'surprise','6':'neutral'}
    if results is not None:
        bboxes = results[0]
        pred_names = results[1]
        face_emotions = results[2]

        exp_list = []
        for img in face_emotions:
            exp_list.append(predict(fer_model,img,graph))
        for index,(x1,y1,x2,y2) in enumerate(bboxes):
            image = cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(image,str(index + 1),(x1,y1-10),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),4)
        emotions = []
        for exp in exp_list:
            emotions.append(EMOTION.get(str(exp)))
        return (pred_names,emotions,image)
    else:
        return -1
def fer_recognition(image, dataset_emb, names_list, face_detect, face_net):
    # 获取 判断标识 bounding_box crop_image
    bboxes, landmarks = face_detect.detect_face(image)
    bboxes, landmarks = face_detect.get_square_bboxes(bboxes, landmarks, fixed="height")
    if bboxes == []:
        return None

    face_emotions = image_processing.get_bboxes_image(image, bboxes, 48,48)
    face_images = image_processing.get_bboxes_image(image, bboxes, 160,160)
    face_images = image_processing.get_prewhiten_images(face_images)

    # 人脸识别部分LY
    pred_emb = face_net.get_embedding(face_images)
    pred_name, pred_score = compare_embadding(pred_emb, dataset_emb, names_list)

    return [bboxes,pred_name,face_emotions]

def predict(fer_model,img,graph):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = np.expand_dims(img,axis=0)
    img = img.reshape(1,48,48,1)
    with graph.as_default():
        classes = fer_model.predict(img)
    classes = classes.tolist()
    predict_results = []
    for i in range(len(classes[0])):
        predict_results.append(classes[0][i])
    max_value = max(predict_results)
    index = predict_results.index(max_value)
    return index

def load_model(weights_path):
    '''
    函数功能:加载表情识别模型
    参数:权重路径
    返回值:模型
    '''
    model = keras.models.Sequential([
    keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(48,48,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.2),

    keras.layers.Flatten(),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(7,activation='softmax')
    ])

    model.load_weights(weights_path)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def load_dataset(dataset_path, filename):
    '''
    加载人脸数据库
    :param dataset_path: embedding.npy文件（faceEmbedding.npy）
    :param filename: labels文件路径路径（name.txt）
    :return:
    '''
    embeddings = np.load(dataset_path)
    names_list = file_processing.read_data(filename, split=None, convertNum=False)

    return embeddings, names_list

def compare_embadding(pred_emb, dataset_emb, names_list, threshold=0.70):
    # 为bounding_box 匹配标签
    pred_num = len(pred_emb)
    dataset_num = len(dataset_emb)
    pred_name = []
    pred_score = []
    for i in range(pred_num):
        dist_list = []
        for j in range(dataset_num):
            dist = np.sqrt(np.sum(np.square(np.subtract(pred_emb[i, :], dataset_emb[j, :]))))
            dist_list.append(dist)
        min_value = min(dist_list)
        pred_score.append(min_value)
        if (min_value > threshold):
            pred_name.append('unknow')

        else:
            pred_name.append(names_list[dist_list.index(min_value)])
    return pred_name, pred_score

def recognition(image, dataset_emb, names_list, face_detect, face_net):
    # 获取 判断标识 bounding_box crop_image
    bboxes, landmarks = face_detect.detect_face(image)
    bboxes, landmarks = face_detect.get_square_bboxes(bboxes, landmarks, fixed="height")
    if bboxes == []:
        return None

    face_images = image_processing.get_bboxes_image(image, bboxes, 160, 160)
    face_images = image_processing.get_prewhiten_images(face_images)

    # 人脸识别部分LY
    pred_emb = face_net.get_embedding(face_images)
    pred_name, pred_score = compare_embadding(pred_emb, dataset_emb, names_list)

    return [bboxes,pred_name]

def opencv_face_recognizer(image,faces,recognizer):
    results = []
    if len(faces) > 0:
        cropped_images = []
        for (x,y,w,h) in faces:
            cropped_images.append(image[y:y+h,x:x+w])

        for cropped_image in cropped_images:
            cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
            results.append(recognizer.predict(cropped_image))

    return results


def opencv_face_detector(image):
    '''
    函数功能:利用opencv模块,检测图片中的人脸
    参数:图片(numpy)
    返回值:框出人脸的图片,人脸个数
    '''
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.15,minNeighbors=5,minSize=(5,5))
    for index,(x, y, w, h) in enumerate(faces):
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image,str(index + 1),(x,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),4)
    return (image,faces)

def get_train_data():
    '''
    函数功能:人脸信息处理
    参数:无
    返回值:人脸(numpy),标签,标签对应人名的字典
    '''
    names_dir = os.listdir('images')
    train_labels = []
    images = []
    train_images = []
    for index,name_dir in enumerate(names_dir):
        name_dir = os.path.join('images',name_dir)
        images_list = os.listdir(name_dir)
        for image in images_list:
            img_path = os.path.join(name_dir,image)
            images.append(img_path)
            train_labels.append(index)

    #把数字对应到人名
    labels = [i for i in range(len(names_dir))]
    name_dict = dict(list(zip(labels,names_dir)))

    for image in images:
        train_images.append(cv2.imread(image,cv2.IMREAD_GRAYSCALE))

    return (train_images,np.array(train_labels),name_dict)


def get_current_radio_combobox_state(mainwindow):
    '''
    函数功能:获取当前radio 和 combobox 的组合状态
    参数:窗口实例
    返回值:各自对应的组合码,失败返回-1
    '''
    radio_text = get_current_radio_text(mainwindow)
    combobox_text = get_current_combobox_text(mainwindow)

    if radio_text == '检测' and combobox_text == 'Haar检测器':
        return 0
    elif radio_text == '检测' and combobox_text == 'MTCNN':
        return 1
    elif radio_text == '识别' and combobox_text == 'LBPH识别':
        return 2
    elif radio_text == '识别' and combobox_text == 'FaceNet':
        return 3
    elif radio_text == '识别' and combobox_text == '表情识别':
        return 4
    else:
        return -1

def get_current_radio_text(mainwindow):
    '''
    函数功能:获取处于选中状态的 radio 文本信息
    参数:窗口实例
    返回值:处于选中状态的 radio 文本信息字符串
    '''
    if mainwindow.radioButton.isChecked():      #这是一个方法,不是变量!!!
        return mainwindow.radioButton.text()
    else:
        return mainwindow.radioButton_2.text()

def get_current_combobox_text(mainwindow):
    '''
    函数功能:获取处于选中状态的 combobox 文本信息
    参数:窗口实例
    返回值:处于选中状态的 combobox 文本信息字符串
    '''
    return mainwindow.comboBox.currentText()


