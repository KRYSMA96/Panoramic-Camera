#coding by Krystian Mao

import cv2
import yaml
from matplotlib.pyplot import imshow
import numpy as np

filepath = 'D:\\BaiduNetdiskDownload\\calibration\\calibration.yaml'

file_data = cv2.FileStorage(filepath,cv2.FILE_STORAGE_READ)
fn = file_data.getNode("intri_camera1")
fd = file_data.getNode("distort_camera1")

print(fn.mat())
print(fd.mat())