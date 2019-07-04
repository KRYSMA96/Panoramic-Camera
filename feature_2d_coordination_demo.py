import cv2
import yaml
import numpy as np

WIDTH = 640
HEIGHT = 480


def parse_yaml_from_name(param_name, yaml_path='/home/xburner/桌面/Smart_city_imgTest/calibration.yaml'):
    ''' 解析相机标定参数文件 '''

    with open(yaml_path, encoding='utf=8', mode='r') as conf:
        cameral_param_dict = yaml.load(conf)

        rows = cameral_param_dict[param_name]['rows']
        cols = cameral_param_dict[param_name]['cols']
        matrix_list = cameral_param_dict[param_name]['data']

        matrix = np.array(matrix_list).reshape((rows, cols))

    return matrix


if __name__ == '__main__':
    img1 = cv2.imread('/home/xburner/桌面/2.jpg')#drive/smart_city/train_data/scene1_jiading_lib_training/PIC_20190522_100025/origin_1.jpg
    resize_img = cv2.resize(img1, (WIDTH, HEIGHT), cv2.INTER_LINEAR)

    intrimatrix = parse_yaml_from_name('intri_camera1')
    distcoeff = parse_yaml_from_name('distort_camera1')

    # 矫正畸变图片
    undistored_img = cv2.fisheye.undistortImage(
        resize_img, K=intrimatrix, D=distcoeff, Knew=intrimatrix)
    # 图片灰度化
    undistored_gray_img = cv2.cvtColor(undistored_img, cv2.COLOR_BGR2GRAY)

    # 提取灰度图像中的SIFT特征点
    detector = cv2.xfeatures2d.SIFT_create()
    keypoints = detector.detect(undistored_gray_img, None)

    # 将特征点绘制到原始彩色图片中
    cv2.drawKeypoints(undistored_gray_img, keypoints, undistored_img)

    # 将KeyPoint格式数据中的xy坐标提取出来。
    points2f = cv2.KeyPoint_convert(keypoints)

    print(points2f)

    cv2.imshow('img', undistored_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 参考
# 1、python opencv
# SIFT，获取特征点的坐标位置：https://www.cnblogs.com/Edison25/p/9921132.html
