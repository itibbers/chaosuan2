import os
import pandas
import numpy as np
from utils import save_data, save_data_without_header

def take_the_cute(data):
    data = np.array(data)
    data_len = len(data)
    start = 0
    for i in range(data_len - 1):
        p1 = data[i][1]
        p2 = data[i + 1][1]
        if p2 - p1 < -1:
            start = i + 1
            break
    timelen = data_len - start
    return start, timelen

# 将四元数处理为xyz
# input:data为list n*m
# output: data_xzy为2维数组
def q4toe2(data):
    data = np.array(data[::3])
    num = len(data)
    data_xyz = np.zeros((num, 3))
    for i in range(num):
        [qx,qy,qz,qw] = data[i][2:6]
        qx = np.float(qx)
        qy = np.float(qy)
        qz = np.float(qz)
        qw = np.float(qw)
        x = 2*qx*qz+2*qy*qw
        y = 2*qy*qz-2*qx*qw
        z = 1-2*(qx**2)-2*(qy**2)
        data_xyz[i][0:3] = [x,y,z]
    return data_xyz


# 将xzy转为经纬度
def xyz2lon_lat(data):
    data = np.array(data[::2])
    num = len(data)
    data_lon_lat = np.zeros((num, 2)) # n*2
    for i in range(num):
        [x,y,z] = data[i]
        theta = np.arctan2(y,x) # return [-pi,pi] 经度
        phi = np.arctan2(np.sqrt(x**2+y**2),z) # return [0, pi] 纬度
        data_lon_lat[i][0] = theta/np.pi*180 # return [-180,180] 经度
        data_lon_lat[i][1] = phi/np.pi*180 # return [0, 180] 纬度
    return data_lon_lat


user_num = 48
video_num = 9
for video_id in range(video_num):
    # video_min = 999999999
    # for user_id in range(1, user_num + 1):
    #     input_path = './data/Experiment_1/'+ str(user_id) + '/video_' + str(video_id) + '.csv'
    #     data = pandas.read_csv(input_path)
    #     start, timelen = take_the_cute(data)
    #     video_min = min(video_min, timelen)

    for user_id in range(1, user_num + 1):
        # 原始数据 to xyz
        input_path = './data/Experiment_1/'+ str(user_id) + '/video_' + str(video_id) + '.csv'
        print(input_path)

        data = pandas.read_csv(input_path)
        # start, timelen = take_the_cute(data)
        # data = data[start:video_min]
        data_xyz = q4toe2(data)

        out_path = './data/data_xyz/' + str(user_id) + '/video_' + str(video_id) + '.csv'
        save_data(data_xyz, out_path)

        # xyz to lon_lat
        input_path = './data/data_xyz/'+ str(user_id) + '/video_' + str(video_id) + '.csv'
        data = pandas.read_csv(input_path)
        data_lon_lat = xyz2lon_lat(data)

        out_path = './data/data_lon_lat/' + str(user_id) + '/video_' + str(video_id) + '.csv'
        save_data_without_header(data_lon_lat, out_path)
