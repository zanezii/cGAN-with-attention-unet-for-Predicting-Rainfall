#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Author   ：Zane
# @Mail     : zanezii@foxmail.com
# @Date     ：2023/5/19 16:52 
# @File     ：utils.py
# @Description :
import sys
import math
import pickle
import numpy as np


# TODO: 评估参数计算
def array_2_matrix(arr: np.array):
    return np.matrix(arr)


def matrix_multi(A: np.matrix, B=None):
    if B is None:
        return A * A.T
    elif isinstance(B, np.matrix) and A.shape == B.shape:
        return A * B.T
    elif isinstance(B, np.matrix) and A.shape != B.shape:
        return A * B
    else:
        print("matrix_multi error")
        sys.exit()


def calc_R(O, P, O_AVE, P_AVE, N=0):
    '''
    :param O: Observe Data of shape(times, area), where area is 128*128
    :param P: Predict Data of shape(times, area), where area is 128*128
    :param O_AVE: average of Observe Data
    :param P_AVE: average of Predict Data
    :param N: num of times of Predict or Observe
    :return: Pearson correlation coefficient,线性相关(-1,1)
    '''
    eq_up = [[0]]
    eq_downl = [[0]]
    eq_downr = [[0]]
    if N == 0:
        N = len(O)
    # 计算等式
    for i in range(N):
        eq_up += matrix_multi(array_2_matrix(O[i] - O_AVE), array_2_matrix(P[i] - P_AVE))
        eq_downl += matrix_multi(array_2_matrix(O[i] - O_AVE))
        eq_downr += matrix_multi(array_2_matrix(P[i] - P_AVE))
    # 1*1的矩阵转数字
    eq_up = float(eq_up[0][0])
    eq_downl = float(eq_downl[0][0])
    eq_downr = float(eq_downr[0][0])

    return eq_up / math.sqrt(eq_downl * eq_downr)


def calc_RMSE(O, P, N=0):
    '''
    :param O: Observe Data of shape(times, area), where area is 128*128
    :param P: Predict Data of shape(times, area), where area is 128*128
    :param N: num of times of Predict or Observe
    :return: root mean square error,标准误差小，测量的可靠性大一些，反之，测量就不大可靠
    '''
    eq_up = [[0]]
    if N == 0:
        N = len(O)
    # 计算等式
    for i in range(N):
        eq_up += matrix_multi(array_2_matrix(O[i] - P[i]))
    # 1*1的矩阵转数字
    eq_up = float(eq_up[0][0])

    return math.sqrt(eq_up / N)


def calc_NSE(O, P, O_AVE, N=0):
    '''
    :param O: Observe Data of shape(times, area), where area is 128*128
    :param P: Predict Data of shape(times, area), where area is 128*128
    :param O_AVE: average of Observe Data
    :param N: num of times of Predict or Observe
    :return: Nash-Sutcliffe efficiency coefficient,验证水文模型模拟结果的好坏(-infi,1),E取值为负无穷至1，
            E接近1，表示模式质量好，模型可信度高；E接近0，表示模拟结果接近观测值的平均值水平，即总体结果可信，但过程模拟误差大；E远远小于0，则模型是不可信的
    '''
    eq_up = [[0]]
    eq_down = [[0]]
    if N == 0:
        N = len(O)
    # 计算等式
    for i in range(N):
        eq_up += matrix_multi(array_2_matrix(O[i] - P[i]))
        eq_down += matrix_multi(array_2_matrix(O[i] - O_AVE))
    # 1*1矩阵转数字
    eq_up = float(eq_up[0][0])
    eq_down = float(eq_down[0][0])

    return 1 - eq_up / eq_down


def calc_CSI(O, P, N=0, area=128 * 128, threshold=0.1):
    '''
    :param O: Observe Data of shape(times, area), where area is 128*128
    :param P: Predict Data of shape(times, area), where area is 128*128
    :param N: num of times of Predict or Observe
    :param area: area 128*128
    :param threshold: intensity thresholds, mm/h
    :return:
        |                |                     |          **Observation**              |
        | -------------- | ------------------- | --------------- | ------------------- |
        |                |                     | Event detected  | Event  not detected |
        | **Prediction** | Event detected      | *Hit*           | *False alarm*       |
        |                | Event  not detected | *Miss*          | *Correct non-event* |

    '''
    hits = 0
    misses = 0
    false_alarms = 0
    if N == 0:
        N = len(O)
    # 计算等式
    for i in range(N):
        for j in range(area):
            obs_v = float(O[i][j])
            pre_v = float(P[i][j])
            if obs_v > threshold and pre_v > threshold:
                hits += 1
            elif obs_v > threshold >= pre_v:
                misses += 1
            elif obs_v <= threshold < pre_v:
                false_alarms += 1
    CSI = hits / (hits + misses + false_alarms)
    return CSI


def calc_FSS(O,P,N,neighborhood=1, threshold=0.1):
    from pysteps import verification
    fss = verification.get_method("FSS")
    # print(O.shape,P.shape)
    # Calculate the FSS for every lead time and all predefined scales.
    scores = []
    for n in range(N):
        score = fss(P[n,:, :, 0], O[n, :, :, 0], threshold, neighborhood)
        if np.isnan(score):
            print(score)
        else:
            scores.append(score)



    return sum(scores)/len(scores)


# TODO: 降雨量（对数单位）与（mm/r）转换
# calculate rain rate R(mm/hr) from dBR
# 替换数组的值
def inverse_dBR_2_R(dBR, threshold=-10):
    if type(dBR) == np.ndarray:  # R<0.1mm/h 设置为0
        dBR[dBR < threshold] = 0
        return dBR
    elif dBR < threshold:
        return 0
    else:
        return 10.0 ** (dBR / 10.0)


# TODO: 归一和逆归一转换
def basic_normalized(data):
    np_min, np_max = np.load("example1/min_max_scale_new.npy")
    return (data - np_min)/(np_max - np_min)


def basic_inverse_normalized(data):
    np_min, np_max = np.load("example1/min_max_scale_new.npy")
    return data * (np_max - np_min) + np_min


def predict_output_scaling(data_input, data_output, proportion=1):
    # range_input = [np.squeeze(Prediction_input[:, :, :, 3]).min(), np.squeeze(Prediction_input[:, :, :, 3]).max()]
    range_input = [np.squeeze(data_input[:, :, :, 3]).min() * proportion,
                   np.squeeze(data_input[:, :, :, 3]).max() * proportion]
    range_output = [np.squeeze(data_output[0, :, :, 0]).min(), np.squeeze(data_output[0, :, :, 0]).max()]
    # print(range_input), print(range_output)

    # OLD PERCENT = (x - OLD MIN) / (OLD MAX - OLD MIN)
    # NEW X = ((NEW MAX - NEW MIN) * OLD PERCENT) + NEW MIN
    output_percent = (data_output - range_output[0]) / (range_output[1] - range_output[0])
    New_Prediction = ((range_input[1] - range_input[0]) * output_percent) + range_input[0]
    return New_Prediction


# TODO: 序列化与非序列化
def serialization_data(data, path='x.pkl'):
    f = open(path, "wb")  # 打开文件x.pkl,写入二进制
    pickle.dump(data, f)


def unserialization_data(path='x.pkl'):
    f = open(path, "rb")
    data = pickle.load(f)
    return data

