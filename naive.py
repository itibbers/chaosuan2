import os
import sys
import csv
import pandas
import math
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Activation, TimeDistributed, Dropout, Lambda, RepeatVector, Input, Reshape
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import json

from utils import *

def Model(train_data, history_step = 30, predict_step = 15):
    train_predict = np.zeros((len(train_data), predict_step, 1))

    for i in range(len(train_data)):
        average = np.mean(train_data[i], 0)
        train_predict[i] = np.tile(average, (predict_step,1))

    return train_predict

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

        for predict_id in range(len(predict_class)):
            predict_step = predict_class[predict_id]
            model_class_name = 'video_{}_{}_h{}p{}_{}'.format(video_id, model_name, history_step, predict_step, col_name)

            # 每个用户 in 20%
            for user_id in range(user_train_size + 1, user_num + 1):
                file_name = '{}/{}/video_{}.csv'.format(config['data_dir_lon_lat'], user_id, video_id)
                data = pandas.read_csv(file_name)
                data = np.array(data)
                data = data[:,col_id] # 取经度或纬度数据

                x_test, y_test = split_sequence(data, history_step, predict_step)
                y_predict = Model(x_test, history_step, predict_step) # n*predic_step*3

                print(y_predict.shape)
                y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
                y_predict = y_predict.reshape(y_test.shape[0], y_test.shape[1])

                # output为2维数组，前n组为实际数据，每组有 预测步长*3 个数据， 后n组为预测数据
                output = np.vstack((y_test, y_predict))
                output_path = './output/video_{}/{}/{}_h{}p{}_{}.csv'.format(video_id, user_id, model_name, history_step, predict_step, col_name)
                save_data_without_header(output, output_path)

