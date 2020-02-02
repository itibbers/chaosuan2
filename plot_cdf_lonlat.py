import pandas
import numpy as np
import math
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from utils import *
import json

config = json.load(open('./config.json'))
history_class = config['history_class']
predict_class = config['predict_class']
model_class = ['lstm_endecoder', 'lstm_standand', 'lstm_m2o', 'mlp_seq', 'lr', 'naive']
model_len = len(model_class)
history_len = len(history_class)
predict_len = len(predict_class)
video_id = config['video_id']

# 数据目录
data_dir = './output/'

# 遍历参数
for i in range(history_len):
    for j in range(predict_len):
        history_step = history_class[i]
        predict_step = predict_class[j]
        params_name = 'h' + str(history_step) + 'p' + str(predict_step) + '_lat'
        out_lon = []
        out_lat = []
        # 遍历模型
        for model_id in range(model_len):
            model_name = model_class[model_id]
            # 每个模型的所有视频数据
            model_lon = []
            model_lat = []

            # 遍历视频
            # for user_id in range(3, 4):
            for user_id in range(9, 11):
                file_name = data_dir + 'video_' + str(video_id) +  '/' + str(user_id) + '/' +  model_name + '_' + params_name + '.csv'
                print(file_name)

                # 读取 data
                data = pandas.read_csv(file_name, header=None)
                data = np.array(data)

                # 处理 data
                num = int(len(data) / 2)
                # print(num)
                test_data = data[0:num, :]
                predict_data = data[num:, :]

                # 计算 data delta
                # delta_lon = np.zeros((num*predict_step))
                delta_data = np.fabs((test_data - predict_data + 180) % 360 - 180)
                rows = delta_data.shape[0]*delta_data.shape[1]
                # delta_lon = np.reshape(delta_data,(rows))
                delta_lat = np.reshape(delta_data,(rows))

                # model_lon.extend(delta_lon)
                model_lat.extend(delta_lat)

            # 每个参数输出结果
            # out_lon.append(model_lon)
            out_lat.append(model_lat)

        out_path = data_dir + 'cdf/' + model_name + '/' + params_name + '.csv'
        save_data(out_lat, out_path)

        font = {
            'family': 'Times New Roman',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16
        }
        for model_id in range(model_len):
            model_name = model_class[model_id]
            ecdf = sm.distributions.ECDF(out_lat[model_id])
            x = np.linspace(0, 200)
            y = ecdf(x)
            plt.ylim((0.8, 1.0))
            plt.plot(x, y, linewidth='1', label=model_name)
            plt.xlabel('xxx', fontdict=font)
            plt.ylabel('yyy', fontdict=font)
            plt.title('ECDF ' + params_name, fontdict=font)
            plt.legend(loc='upper right', shadow=True)
        file_path = data_dir + 'cdf/' + params_name + '.png'
        plt.savefig(file_path, dpi=200)
        plt.show()
