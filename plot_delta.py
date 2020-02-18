# plot_delta
import pandas
import numpy as np
import math
from math import radians, cos, sin, asin, sqrt
import json
import matplotlib.pyplot as plt
from utils import *
import multiprocessing

config = json.load(open('./config.json'))
history_class = config['history_class']
predict_class = config['predict_class']
## predict_class = [1,2]
model_class = ['lstm_m2o', 'lr2', 'mlp_seq', 'naive']
model_class_dummy = ['naive', 'lr2', 'mlp_seq', 'lstm_m2o']
model_len = len(model_class)
model_name = 'naive'
history_len = len(history_class)
predict_len = len(predict_class)
video_id = config['video_id']
col_name = config['col_class'][0]
# 数据目录
data_dir = './output'

tiles = [
    [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116],
    [201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216],
    [301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316],
    [401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416],
    [501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516],
    [601,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416],
    [701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716],
    [801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816],
]

"""
@desc 经纬转tile数组
@param 经度、纬度、tile数组，视口宽度，视口高度
0-360, 0-180, 16*8, 6, 4
一个格子大小：22.5x22.5
@return tileArray
"""
def ll2tileArray(x, y, vx = 135, vy = 90):
    lenx = vx/2
    leny = vy/2
    x1 = math.floor((x - lenx) / 22.5)
    x2 = math.ceil((x + lenx) / 22.5)
    y1 = math.floor((y - leny) / 22.5)
    y2 = math.ceil((y + leny) / 22.5)

    res = []
    for i in range(y1, y2):
        ii = i % 8 # tiles height
        t = []
        for j in range(x1, x2):
            jj = j % 16 # tiles width
            t.append(tiles[ii][jj])
        res.append(t)
    return np.array(res)

# print(ll2tileArray(67.5, 45))
# exit()

for user_id in range(9, 11):
# def oneUser(user_id):
    res = []
    for i in range(history_len):
        for j in range(predict_len):
            history_step = history_class[i]
            predict_step = predict_class[j]

            # 经度
            params_name = 'h{}p{}_lon'.format(history_step, predict_step)
            file_name = '{}/video_{}/{}/{}_{}.csv'.format(data_dir, video_id, user_id, model_name, params_name)
            print(file_name)
            data = pandas.read_csv(file_name, header=None)
            data = np.array(data)
            # print(len(data))

            num = int(len(data) / 2)
            # 经度数据
            lon_test_data = data[0:num, :]
            lon_predict_data = data[num:, :]

            # 纬度
            params_name = 'h{}p{}_lat'.format(history_step, predict_step)
            file_name = '{}/video_{}/{}/{}_{}.csv'.format(data_dir, video_id, user_id, model_name, params_name)
            print(file_name)
            data = pandas.read_csv(file_name, header=None)
            data = np.array(data)
            # print(len(data))

            num = int(len(data) / 2)
            # 纬度数据
            lat_test_data = data[0:num, :]
            lat_predict_data = data[num:, :]

            avg = []
            # ?
            # 有n组，每组有predict_step个数据
            # n = int(num / predict_step)
            # print(n)
            for k in range(len(lon_test_data)):
                # ?
                # kid = (k + 1) * predict_step - 1
                # kid = k
                for kid in range(len(lon_test_data[0])):
                    # ?
                    # # 实际经纬度
                    # lon1 = lon_test_data[kid][len(lon_test_data[0])-1]
                    # lat1 = lat_test_data[kid][len(lat_test_data[0])-1]
                    # # 预测经纬度
                    # lon2 = lon_predict_data[kid][len(lon_test_data[0])-1]
                    # lat2 = lat_predict_data[kid][len(lat_test_data[0])-1]
                    # #print(lon1,lat1,lon2,lat2)

                    # 实际经纬度
                    lon1 = lon_test_data[k][kid]
                    lat1 = lat_test_data[k][kid]
                    # 预测经纬度
                    lon2 = lon_predict_data[k][kid]
                    lat2 = lat_predict_data[k][kid]
                    #print(lon1,lat1,lon2,lat2)

                    tile1 = ll2tileArray(lon1, lat1)
                    tile2 = ll2tileArray(lon2, lat2)
                    # print(tile1)
                    # print(tile2)
                    # 交集元素个数
                    inters = np.intersect1d(tile1, tile2)
                    real = tile1.reshape(tile1.shape[0] * tile1.shape[1])
                    accuracy = len(inters) / len(real)
                    # print(inters)
                    # print(real)
                    # print(accuracy)
                    avg.append(accuracy)
            print(np.mean(avg))
            res.append(np.mean(avg))
    out_path = '{}/video_{}/delta_{}.csv'.format(data_dir, video_id, user_id)
    out = []
    out.append(res)
    save_data_without_header(out, out_path)
    plt.plot(predict_class, res, label=model_name + '_' + str(user_id))

# 计算其它用户
# others = []
# for user_id in range(39, 49):

def runOther(user_id):
    res = []
    for i in range(history_len):
        for j in range(predict_len):
            history_step = history_class[i]
            predict_step = predict_class[j]

            # 经度
            params_name = 'h{}p{}_lon'.format(history_step, predict_step)
            file_name = '{}/video_{}/{}/{}_{}.csv'.format(data_dir, video_id, user_id, model_name, params_name)
            print(file_name)
            data = pandas.read_csv(file_name, header=None)
            data = np.array(data)
            # print(len(data))

            num = int(len(data) / 2)
            # 经度数据
            lon_test_data = data[0:num, :]
            lon_predict_data = data[num:, :]

            # 纬度
            params_name = 'h{}p{}_lat'.format(history_step, predict_step)
            file_name = '{}/video_{}/{}/{}_{}.csv'.format(data_dir, video_id, user_id, model_name, params_name)
            print(file_name)
            data = pandas.read_csv(file_name, header=None)
            data = np.array(data)
            # print(len(data))

            num = int(len(data) / 2)
            # 纬度数据
            lat_test_data = data[0:num, :]
            lat_predict_data = data[num:, :]

            avg = []
            # ?
            # 有n组，每组有predict_step个数据
            # n = int(num / predict_step)
            # print(n)
            for k in range(len(lon_test_data)):
                # ?
                # kid = (k + 1) * predict_step - 1
                # kid = k
                for kid in range(len(lon_test_data[0])):
                    # ?
                    # # 实际经纬度
                    # lon1 = lon_test_data[kid][len(lon_test_data[0])-1]
                    # lat1 = lat_test_data[kid][len(lat_test_data[0])-1]
                    # # 预测经纬度
                    # lon2 = lon_predict_data[kid][len(lon_test_data[0])-1]
                    # lat2 = lat_predict_data[kid][len(lat_test_data[0])-1]
                    # #print(lon1,lat1,lon2,lat2)

                    # 实际经纬度
                    lon1 = lon_test_data[k][kid]
                    lat1 = lat_test_data[k][kid]
                    # 预测经纬度
                    lon2 = lon_predict_data[k][kid]
                    lat2 = lat_predict_data[k][kid]
                    #print(lon1,lat1,lon2,lat2)

                    tile1 = ll2tileArray(lon1, lat1)
                    tile2 = ll2tileArray(lon2, lat2)
                    # print(tile1)
                    # print(tile2)
                    # 交集元素个数
                    inters = np.intersect1d(tile1, tile2)
                    real = tile1.reshape(tile1.shape[0] * tile1.shape[1])
                    accuracy = len(inters) / len(real)
                    # print(inters)
                    # print(real)
                    # print(accuracy)
                    avg.append(accuracy)
            print(np.mean(avg))
            res.append(np.mean(avg))
    # others.append(res)
    return res

# Get all cores
cores = multiprocessing.cpu_count()

# 单用户
# users = range(9, 11)
# poolOne = multiprocessing.Pool(processes=cores)
# poolOne.map(oneUser, users)

if 1:
    users = range(39, 49)
    # start a pool
    pool = multiprocessing.Pool(processes=cores)
    # do parallel calculate
    others = pool.map(runOther, users)

    othersAvg = np.mean(others, axis=0)
    out_path = '{}/video_{}/others.csv'.format(data_dir, video_id)
    out = []
    out.append(othersAvg)
    save_data_without_header(out, out_path)
else:
    othersAvg = pandas.read_csv('./output/video_' + str(video_id) + '/others.csv', header=None)
    othersAvg = np.array(othersAvg)[0]

plt.plot(predict_class, othersAvg, label=model_name + '_39-48')

plt.legend()
plt.show()

