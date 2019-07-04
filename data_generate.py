#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:10:33 2019

@author: xburner
"""

import random
import os
from imutils import paths
import cv2
import yaml
import re
import csv
import numpy as np


WIDTH = 640
HEIGHT = 480


def parse_yaml_from_name(param_name, yaml_path='drive/Smart_City_data/calibration.yaml'):
    ''' 解析相机标定参数文件 '''

    with open(yaml_path, encoding='utf=8', mode='r') as conf:
        cameral_param_dict = yaml.load(conf)

        rows = cameral_param_dict[param_name]['rows']
        cols = cameral_param_dict[param_name]['cols']
        matrix_list = cameral_param_dict[param_name]['data']

        matrix = np.array(matrix_list).reshape((rows, cols))

    return matrix



def load_data(path):
    #===========================================================================
    #csv文件读取
    f_list = os.listdir(path)
    
    for i in f_list:
        
        if os.path.splitext(i)[1] == '.csv':
            
            csv_path = path + '/'+ i
    
    csvFile = open(csv_path,'r')
    
    c_reader = csv.reader(csvFile)
    
    result = {}
    
    for item in c_reader:
        
        if c_reader.line_num == 1:
            
            continue
        
        coordinates = item[1:]
        
        result[item[0]] = coordinates#键值对
        
        np.save("drive/Smart_City_data/csv.npy",result)
    #===========================================================================
    #图片处理
    image_paths = sorted(list(paths.list_images(path)))#读取path文件路径下所有子文件路径
    
    pattern = re.compile(r'drive/smart_city/train_data/scene1_jiading_lib_training/PIC_\d+_\d+/origin_\d.jpg')
    
    origin_paths = pattern.findall(",".join(image_paths))#匹配所有的origin图片路径
    
    random.seed(0)
    
    random.shuffle(origin_paths)
    
    i = 0
    
    npy_paths = []
    
    for img_path in origin_paths:
        
        img1 = cv2.imread(img_path)
        
        pattern = re.compile(r'drive/smart_city/train_data/scene1_jiading_lib_training/(PIC_\d+_\d+)/origin_(\d).jpg')
        
        numbers = pattern.findall(img_path)
        #print(numbers)
        csv_dir = numbers[0][0]#csv文件里面的label坐标对应的文件夹编码
        
        origin_num = numbers[0][1]#origin镜头的编码
        
        #=======================================================================
        #特征点提取
                        
        resize_img = cv2.resize(img1, (WIDTH, HEIGHT), cv2.INTER_LINEAR)

        intrimatrix = parse_yaml_from_name('intri_camera'+origin_num)#origin_num是字符串
        
        distcoeff = parse_yaml_from_name('distort_camera' +origin_num)

        # 矫正畸变图片
        undistored_img = cv2.fisheye.undistortImage(
            resize_img, K=intrimatrix, D=distcoeff, Knew=intrimatrix)
        
        # 图片灰度化
        undistored_gray_img = cv2.cvtColor(undistored_img, cv2.COLOR_BGR2GRAY)

            
        # 提取灰度图像中的SIFT特征点
        detector = cv2.xfeatures2d.SIFT_create()
        
        keypoints = detector.detect(undistored_gray_img, None)

        # 将特征点绘制到原始彩色图片中
        #cv2.drawKeypoints(undistored_gray_img, keypoints, undistored_img)

        # 将KeyPoint格式数据中的xy坐标提取出来。
        points2f = cv2.KeyPoint_convert(keypoints)

        #print(points2f)

        points = points2f.reshape([1,-1])
        #print(points)
        #======================================================================
        #将数据与label拼接起来
        label = np.array(result[csv_dir],dtype = np.float32).reshape([1,3])#将列表包装成numpy数组方便操作
        #print(label)
        train_sample = np.concatenate((points,label),axis = 1)#横向合并
        
        #print(points)
        
        current_npy_path = "drive/Smart_City_data/FeatruePoints_"+csv_dir+"_origin_"+origin_num+".npy"
        
        npy_paths.append(current_npy_path)
        
        np.save(current_npy_path,train_sample)
        
        i += 1
        
        print(i)
        
    #==========================================================================
    #将所有的npy全部拼接起来
    current_npy_length = 0
    
    max_npy_length = 0
    #找到最大npy长度
    
    p = 0
    
    for each_path in npy_paths:
        
        current_npy = np.load(each_path)
        
        current_npy_length = current_npy.shape[1]
        
        if current_npy_length > max_npy_length:
            
            max_npy_length = current_npy_length
        
        p += 1
        
        print(p)
            
    print(max_npy_length)#检验最长npy长度
    
    #将每一条npy数据都扩充到最大长度
    data_all = np.zeros((1,max_npy_length))
    print(data_all.shape)
    
    k = 0
    
    for each_path in npy_paths:
        
        current_npy = np.load(each_path)
        
        if current_npy.shape[1] < max_npy_length:
            
            sub_length = max_npy_length - current_npy.shape[1]
            #print(current_npy[:-3].shape)
            #print(current_npy.shape)
            #print(np.zeros((1,sub_length)).shape)
            temp = np.concatenate((current_npy[:,:-3],np.zeros((1,sub_length))),axis = 1)
            #print(temp.shape)
            current_npy = np.concatenate((temp,current_npy[:,-3:]),axis = 1)#横向
            
            #print(current_npy.shape)
        
        data_all = np.concatenate((data_all,current_npy),axis = 0)#纵向
        
        k += 1
        
        print(k)
        
    data_all = data_all[1:,:]#去除第一行的全零样本
    
    np.save("drive/Smart_City_data/data_all.npy",data_all)
        
        
if __name__ == '__main__':
    
    load_data('drive/smart_city/train_data/scene1_jiading_lib_training')#输入场景路径

