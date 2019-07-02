#coding by Krystian Mao

import cv2
import yaml
from matplotlib.pyplot import imshow
import numpy as np

filepath = 'D:\\BaiduNetdiskDownload\\calibration\\calibration.yaml'

def undistort():
    skip_lines = 0
    with open(filepath) as infile:
        for i in range(skip_lines):
            _ = infile.readline()
        #file_data = yaml.load(infile)
        file_data = cv2.FileStorage(filepath,cv2.FILE_STORAGE_READ)

    #replace 3 lines withoutput in the calibration step
    
    #K = np.array()
    #D = np.array()

    intri_matrix = file_data.getNode("intri_camera6")
    K = intri_matrix.mat()
    
    Distortion_matrix = file_data.getNode("distort_camera6")
    D = Distortion_matrix.mat()

    #K = np.asarray([[fu,0,pu],[0,fv,pv],[0,0,1]]) #K(3,3)
    #D = np.asarray(file_data['cam0']['distortion_coeffs'])  
    

    img = cv2.imread("D:/Panoramic-Camera-Fenghaoyu/origin_1.jpg")
    
    h,w = img.shape[:2]
    DIM = (h,w)


    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    #cv2.fisheye.undistortImage(img,undistorted_img,K,D,K,DIM)
    #error: (-215:Assertion failed) (K.depth() == 5 || K.depth() == 6) && (D.depth() == 5 || D.depth() == 6) in function 'cv::fisheye::initUndistortRectifyMap'
    cv2.imwrite("undistort_origin_1.jpg",undistorted_img)

if __name__ == '__main__':
    undistort()


