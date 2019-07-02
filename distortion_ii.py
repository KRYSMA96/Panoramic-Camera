import cv2
import yaml
import numpy as np

WIDTH = 640
HEIGHT = 480
DIM = (640,480)

filepath = 'D:/BaiduNetdiskDownload/calibration/calibration.yaml'

skip_lines = 0
with open(filepath) as f:
    for i in range(skip_lines):
        _ = f.readline()
    f_data = cv2.FileStorage(filepath,cv2.FILE_STORAGE_READ)



if __name__ == '__main__':
    img1 = cv2.imread('D:/Panoramic-Camera-Fenghaoyu/origin_1.jpg')
    #resize_img = cv2.resize(img1, (WIDTH, HEIGHT), cv2.INTER_LINEAR)
    #resize_img = cv2.resize(img1,(640,480),cv2.INTER_LINEAR)
    img1 = cv2.resize(img1, DIM, cv2.INTER_LINEAR)
    #cv2.resize(src = img1, dsize = (640, 480), interpolation = cv2.INTER_LINEAR)

    intrimatrix = np.matrix(f_data.getNode("intri_camera1").mat())
    distcoeff = np.matrix(f_data.getNode("distort_camera1").mat())
    print(intrimatrix)


    img = cv2.fisheye.undistortImage(
        img1, 
        K=intrimatrix,
        D=distcoeff, 
        Knew=intrimatrix)
##############################################################

    cv2.imshow('resize_img', img1)
    cv2.imshow('undistorted', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

  