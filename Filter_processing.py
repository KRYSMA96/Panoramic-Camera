#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:38:22 2019

@author: xburner
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
#读取图像
img = cv2.imread('/home/xburner/桌面/Smart_city_imgTest/thumbnail.jpg',0)



#图像高斯滤波
gauss = cv2.GaussianBlur(img,(7,7),2)

#滤波之后使用sobel算子
x = cv2.Sobel(gauss,cv2.CV_16S,1,0) 
y = cv2.Sobel(gauss,cv2.CV_16S,0,1)

absX = cv2.convertScaleAbs(x) # 转回uint8 
absY = cv2.convertScaleAbs(y) 
dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
cv2.imwrite('/home/xburner/桌面/Smart_city_imgTest/gauss_sobel1.jpg',dst)
#====================================================
#傅立叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)#低频部分移到中心
rows,cols = img.shape
crow, ccol = int(rows/2),int(cols/2)

#设置低通滤波器
mask = np.zeros((rows, cols), np.uint8)
mask[crow-100:crow+100, ccol-100:ccol+100] = 1
lowpass = fshift * mask

#设置高通滤波器
fshift[crow - 30:crow + 30,ccol - 30:ccol + 30] = 0#低频部分置0

#傅立叶逆变换
ishift = np.fft.ifftshift(fshift)#高通
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)
low_ishift = np.fft.ifftshift(lowpass)#低通
low_img = np.fft.ifft2(low_ishift)
low_img = np.abs(low_img)
#====================================================
#高通滤波后使用Canny边缘检测
iimg = iimg.astype(np.uint8)
canny_img = cv2.Canny(iimg,40,70)

#高斯滤波后使用拉普拉斯算子
gray_lap = cv2.Laplacian(gauss, cv2.CV_16S, ksize=3)
dst1 = cv2.convertScaleAbs(gray_lap)


#显示滤波之前图像与之后的图像
plt.subplot(221), plt.imshow(img,'gray'),plt.title('Original_Image')
plt.axis('off')
plt.subplot(222),plt.imshow(iimg,'gray'),plt.title('High_pass')
plt.axis('off')
plt.subplot(223),plt.imshow(dst1,'gray'),plt.title('guass_Laplacian')
plt.axis('off')
plt.subplot(224),plt.imshow(dst,'gray'),plt.title('gauss_sobel')
plt.axis('off')
plt.show()    