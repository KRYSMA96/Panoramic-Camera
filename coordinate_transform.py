#coding by Krystian Mao

import numpy as np 
import matplotlib.pyplot as plot 
import cv2

'''
filepath = 'D:/BaiduNetdiskDownload/calibration/calibration.yaml'

skip_lines = 0
with open(filepath) as f:
    for i in range(skip_lines):
        _ = f.readline()
    f_data = cv2.FileStorage(filepath,cv2.FILE_STORAGE_READ)

intri_matrix =  np.matrix(f_data.getNode("intri_camera1").mat())
#print(intri_matrix)

inverse_matrix = intri_matrix.I   #求逆，这个逆矩阵是图像坐标到相机坐标的逆变换,也就是内参矩阵的逆
'''

def cal_camera(u,v):
    f = 0.02  #焦距
    Projection_Matrix = [[f,0,0],    #透视投影矩阵
                        [0,f,0],
                        [0,0,1]]

    u0 = 640/2
    v0 = 480/2


    x = u - u0
    y = v - v0

    #Pixel2Img = [[a,r,u0]             #图像坐标转换为像素坐标矩阵
    #            [0,b,vo]
    #            [0,0,1]]

    #camera_Matrix = [Xc,Yc,Zc]         #相机坐标矩阵
    
    Zc = 1                             #attention这里设置了Zc是1
    Pixel = [u,v,1] * Zc               #像素坐标
    Image_  = [x,y,Zc]                 #图片坐标          

    inverse_Projection =  np.matrix(Projection_Matrix).I
                                       #图像坐标到摄像机坐标的逆操作
    #inverse_Pixel = Pixel2Img.I 
                                       #像素坐标到图像坐标的逆操作

    #camera_Matrix = inverse_Pixel * inverse_Projection * Pixel 
    camera_Matrix = inverse_Projection * np.reshape(Image_ , (3,1))
    return camera_Matrix

if __name__ == "__main__":
    matrix = cal_camera(480,300)
    print(matrix)

