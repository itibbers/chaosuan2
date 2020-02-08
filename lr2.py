import os
import sys
import csv
import pandas
import numpy as np
from sklearn import linear_model

import json

from utils import *

def Model():
    return linear_model.LinearRegression()

model_name = sys.argv[0].replace(".py", "")
config = json.load(open('./config.json'))

video_id = config['video_id']
col_class = config['col_class']
user_num = config['user_num']
validate_percent = config['validate_percent']
user_train_size = int(user_num * validate_percent)
history_class = config['history_class']
predict_class = config['predict_class']


for col_id in range(len(col_class)):
    col_name = col_class[col_id]

    for history_id in range(len(history_class)):
        history_step = history_class[history_id]

        xi = np.linspace(-history_step, -1, history_step)[:,np.newaxis] # -30~-1 等分为30份 xi=[-30,-29,-28,...-1]

        for predict_id in range(len(predict_class)):
            predict_step = predict_class[predict_id]
            print(predict_step)

            x = np.linspace(1,predict_step,predict_step)[:,np.newaxis] # 1~60 等分为60份 预测60个值
            model_class_name = 'video_{}_{}_h{}p{}_{}'.format(video_id, model_name, history_step, predict_step, col_name)
            model = Model()

            # 不用划分训练集测试集，历史30个数用来fit（即训练集），直接预测（测试集）
            for user_id in range(user_num-1, user_num+1):
                file_name = '{}/{}/video_{}.csv'.format(config['data_dir_lon_lat'], user_id, video_id)
                print(file_name)
                data = pandas.read_csv(file_name)
                data = np.array(data)
                data = data[:,col_id] # 取经度或纬度数据
                data = data.reshape((data.shape[0],1))
                y_test = []
                y_predict = []

                for i in range(data.shape[0]-(history_step+predict_step)):
                    yi = data[i : i+history_step] # 历史30个数据
                    y_real = data[i+history_step:i+history_step+predict_step] # 待预测的真实值
                    # print(yi.shape, y_real.shape)
                    model.fit(xi, yi) # 拟合xy曲线
                    y = model.predict(x) # 预测

                    # print(y_real.shape)
                    print(len(y_test))
                    y_test.extend(y_real)
                    y_predict.extend(y)

                y_test = np.array(y_test)
                y_predict = np.array(y_predict)

                y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
                y_predict = y_predict.reshape((y_predict.shape[0], y_predict.shape[1]))
                output = np.vstack((y_test, y_predict))
                output_path = './output/video_{}/{}/{}_h{}p{}_{}.csv'.format(video_id, user_id, model_name, history_step, predict_step, col_name)
                save_data_without_header(output, output_path)







