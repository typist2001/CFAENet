import cv2
import numpy as np
from tqdm import tqdm
import os

src_root = '/home/root803/gfq/RGBD/RGBD-SOD/TrainDataset/GT/'
src = '/home/root803/gfq/RGBD/RGBD-SOD/TrainDataset/edge/'
for image_name in tqdm(os.listdir(src_root)):
    gt = cv2.imread(src_root + image_name)

    # 使用算子

    # gt = cv2.GaussianBlur(gt,(3,3),0)
    # canny = cv2.Canny(gt, 128, 255, apertureSize = 3)     # 调用Canny函数，指定最大和最小阈值，其中apertureSize默认为3。
    # cv2.imwrite(image_name, canny)

    # 形态学：边缘检测
    _, Thr_img = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY)  # 设定红色通道阈值210（阈值影响梯度运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 定义矩形结构元素
    gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)  # 梯度

    cv2.imwrite(src+image_name, gradient)