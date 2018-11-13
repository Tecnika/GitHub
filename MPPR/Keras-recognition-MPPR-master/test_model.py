# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
import os
import sys

import cv2
import numpy as np
import zmq
from keras.models import load_model
from keras.utils import plot_model

from PIL import Image, ImageFilter

#изменение текущего окружения на окружение скрипта
pathProgramm = os.path.dirname(sys.argv[0])
if len(pathProgramm) > 0:
    os.chdir(pathProgramm)

# Запуск сервера
context = zmq.Context(1)
server = context.socket(zmq.REP)
server.bind("tcp://*:5556")

#загрузка модели
print("I: Model loading ...")
try:
    MODEL = load_model("digits_cls.ckpt")
except IOError:
    print("E: No digits_cls")
    exit()
MODEL.summary()

print("I: Server started")

while True:

    request = server.recv()

    try:
        image = Image.open("image.png")  # Открытие изображения
        imageconvert = image.convert('L')
    except IOError:
        server.send(request)
        print("E: No image")
        continue

    width = int(round(float(imageconvert.size[0]+100)))
    height = int(round(float(imageconvert.size[1]+100)))

    # Новое полотно на 50 больше
    newImage = Image.new('L', (width, height), (255))

    newImage.paste(imageconvert, (50, 50))
    newImage.save("image.png")

    try:
        im = cv2.imread("image.png")
    except IOError:
        server.send(request)
        print("E: No image")
        continue
    # Конвертация в серый слой и применение фильтра Гаусса
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Поиск изображения
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    cv2.imwrite('temp_RESIZE.bmp', im_th)

    # Поиск контуров на изображении
    # im_th - исходное изображение
    # RETR_EXTERNAL - режим поиска контуров
    # CHAIN_APPROX_SIMPLE - метод приближения контуров
    _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not ctrs:
        server.send(request)
        print("E: Empty image")
        continue
    
    #print(ctrs)
    try:
        rect = cv2.boundingRect((sorted(ctrs, key=cv2.contourArea, reverse=True))[0])  # получить самый большой контур

        roi = im_th[rect[1] - 10 :rect[1]+rect[3] + 10, rect[0] - 10:rect[0]+rect[2]+ 10]

        old_size = roi.shape[:2]  # old_size is in (height, width) format

        # Подготовить MNIST-изображение
        desired_size = 28

        ratio = float(desired_size)/((max(old_size)))
        new_size = tuple([int(x*ratio) for x in old_size])
        # new_size should be in (width, height) format
        roi = cv2.resize(roi, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1] 
        delta_h = desired_size - new_size[0]  
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0]
        roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        cv2.imwrite('temp_mnist.bmp', roi)

        #создаем одномерный мвассив и отправляем на распознование
        out = MODEL.predict(roi.reshape((1, -1)))
        print(out)
        print(u'Распознаная цифра: ',np.argmax(out))
        request = np.argmax(out)
    except IOError as err:
        print("I/O error: {0}".format(err))
    except ValueError:
        print("Could not convert data to an integer.")
    except cv2.error as cve:
        print("Don't worry, it is OpenCV.", cve)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    server.send(request)

server.close()
context.term()
