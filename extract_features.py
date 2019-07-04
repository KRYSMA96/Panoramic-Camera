import cv2

import pylab as plt

import yaml
import numpy as np
import pandas as pd
import os
import fnmatch
import csv

import time

print(cv2.__version__)

os.chdir('/content/drive/')
# !find / -name 'smart_city'

WIDTH = 640
HEIGHT = 480

PARAM_PATH = 'nn/smart_city/calibration.yaml'
TRAIN_DATA_PATH = 'nn/smart_city/train_data/scene1_jiading_lib_training'
FEATRUE_PATH = 'nn/smart_city/feature_scene1.csv'
LABEL_PATH = os.path.join(TRAIN_DATA_PATH,
                          'scene1_jiading_lib_training_coordinates.csv')


def parse_yaml_from_name(param_name, yaml_path=PARAM_PATH):
    ''' 解析相机标定参数文件 '''

    with open(yaml_path, encoding='utf=8', mode='r') as conf:
        cameral_param_dict = yaml.load(conf)

        rows = cameral_param_dict[param_name]['rows']
        cols = cameral_param_dict[param_name]['cols']
        matrix_list = cameral_param_dict[param_name]['data']

        matrix = np.array(matrix_list).reshape((rows, cols))

    return matrix


def get_fixedsize_gray_img(path):
    '''给定图片路径，返回指定像素的灰度图片'''

    img = cv2.imread(path)
    img = cv2.resize(img, (WIDTH, HEIGHT), cv2.INTER_LINEAR)

    return img


start_time = time.time()

if __name__ == '__main__':
    # 读取标记数据（相机的坐标位置）
    count = 0
    label_data = pd.read_csv(LABEL_PATH)
    labels = label_data.values

    # 默认topdown为True，表示先返回根目录下的文件，然后遍历这些文件中的目录；为False，表示先遍历到最深层的子目录，再返回
    for root, dirs, files in os.walk(TRAIN_DATA_PATH):
        if len(dirs) != 0:
            #       print(dirs)
            # 使按照文件夹的自然排序进行遍历
            dirs.sort()
            continue

#     print(root)
        label = labels[count][1:]
        count += 1

        # 读取全景图片
        thumbnail_img = get_fixedsize_gray_img(
            os.path.join(root, 'thumbnail.jpg'))
        for file in fnmatch.filter(files, '*[1-6].jpg'):
            # 读取部分图片
            distored_img = get_fixedsize_gray_img(os.path.join(root, file))
            # 相机的相对位置
            num = file.split('_')[1][:1]

            intrix_matrix_param = 'intri_camera' + num
            intrimatrix = parse_yaml_from_name(intrix_matrix_param)
#         print(intrimatrix)
            distortion_coeff_param = 'distort_camera' + num
            distcoeff = parse_yaml_from_name(distortion_coeff_param)
#         print(distcoeff)

            undistored_img = cv2.fisheye.undistortImage(distored_img, K=intrimatrix,
                                                        D=distcoeff,
                                                        Knew=intrimatrix)
#         plt.imshow(undistored_img)
#         plt.imshow(thumbnail_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

            # 使用SIFT检测角点
            sift = cv2.xfeatures2d.SIFT_create()
            # 获取关键点和描述符
            kp_part, des_part = sift.detectAndCompute(undistored_img, None)
            kp_thumbnail, des_thumbnail = sift.detectAndCompute(
                thumbnail_img, None)

            # 定义FLANN匹配器
            index_params = dict(algorithm=1, trees=10)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            # 使用KNN算法匹配
            matches = flann.knnMatch(des_part, des_thumbnail, k=2)

            # 去除错误匹配
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            dst_pts = np.float32(
                [kp_thumbnail[m.trainIdx].pt for m in good]).reshape(-1, 2)

            label = np.tile(label, (len(dst_pts), 1))
            dst_pts = np.hstack((dst_pts, label))

            # 将特征写入到csv文件中
            with open(FEATRUE_PATH, mode='a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(dst_pts)

#       break
#     break

end_time = time.time()
print('run time is %ss' % (end_time - start_time))


# 参考
# 1、os.walk参数topdown的作用：https://www.cnblogs.com/xiaxiaoxu/p/9746819.html
# 2、我可以强制python3的os.walk按字母顺序访问目录吗？怎么样？：https://cloud.tencent.com/developer/ask/186840
# 3、Python模块学习 - fnmatch & glob：https://www.cnblogs.com/dachenzi/p/8215584.html
# 4、Pandas中把dataframe和np.array的相互转换：https://blog.csdn.net/weixin_39223665/article/details/79935467
# 5、 Python Pandas : How to add new
# columns：https://thispointer.com/python-pandas-how-to-add-new-columns-in-a-dataframe-using-or-dataframe-assign/
