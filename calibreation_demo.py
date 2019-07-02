import cv2
import yaml
import numpy as np

WIDTH = 640
HEIGHT = 480


def parse_yaml_from_name(param_name, yaml_path='calibration.yaml'):
    ''' 解析相机标定参数文件 '''

    with open(yaml_path, encoding='utf=8', mode='r') as conf:
        cameral_param_dict = yaml.load(conf)

        rows = cameral_param_dict[param_name]['rows']
        cols = cameral_param_dict[param_name]['cols']
        matrix_list = cameral_param_dict[param_name]['data']

        matrix = np.array(matrix_list).reshape((rows, cols))

    return matrix


if __name__ == '__main__':
    img1 = cv2.imread('origin_1.jpg')
    resize_img = cv2.resize(img1, (WIDTH, HEIGHT), cv2.INTER_LINEAR)

    intrimatrix = parse_yaml_from_name('intri_camera1')
    distcoeff = parse_yaml_from_name('distort_camera1')

    # print('intrixmatrix:', intrimatrix)
    # print('distcoeff:', distcoeff)

    # cv2.imwrite('test.jpg', resize_img)

    # new_intrimatrix = intrimatrix.copy()
    # new_intrimatrix[(0, 1), (0, 1)] = 0.8 * new_intrimatrix[(0, 1), (0, 1)]
    # img = cv2.fisheye.undistortImage(
    #     img1, intrimatrix, D=distcoeff, Knew=new_intrimatrix)

    img = cv2.fisheye.undistortImage(
        resize_img, K=intrimatrix, D=distcoeff, Knew=intrimatrix)

    cv2.imshow('resize_img', resize_img)
    cv2.imshow('undistorted', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
