#coding by Krystian Mao

import numpy as np 
import matplotlib.pyplot as plot 
import cv2


def cal_camera(u,v):
    f = 0.02                           #焦距
    Projection_Matrix = [[f,0,0],      #透视投影矩阵
                        [0,f,0],
                        [0,0,1]]

    u0 = 640/2
    v0 = 480/2

    x = (u - u0)/8000
    y = (v - v0)/8000                  #80个像素点1mm

    #Pixel2Img = [[a,r,u0]             #图像坐标转换为像素坐标矩阵
    #            [0,b,vo]
    #            [0,0,1]]

    #camera_Matrix = [Xc,Yc,Zc]        #相机坐标矩阵
    
    Zc = 1                             #attention!!!这里设置了Zc是1
    #Pixel = [u,v,1] * Zc              #像素坐标
    Image_  = [x,y,Zc]                 #图片坐标          

    inverse_Projection =  np.matrix(Projection_Matrix).I
                                       #图像坐标到摄像机坐标的逆操作
    #inverse_Pixel = Pixel2Img.I 
                                       #像素坐标到图像坐标的逆操作

    #camera_Matrix = inverse_Pixel * inverse_Projection * Pixel 
    camera_Matrix = inverse_Projection * np.reshape(Image_ , (3,1))
    return camera_Matrix                


if __name__ == "__main__":
    matrix = cal_camera(400,260)
    print(matrix)


