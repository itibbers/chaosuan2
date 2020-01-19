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
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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

        for predict_id in range(len(predict_class)):
            predict_step = predict_class[predict_id]
            model_class_name = 'video_{}_{}_h{}p{}_{}'.format(video_id, model_name, history_step, predict_step, col_name)
            model = Model()

            x_train_all = []
            y_train_all = []
            # 每个用户 in 80%
            for user_id in range(1, user_train_size + 1):
                file_name = '{}/{}/video_{}.csv'.format(config['data_dir_lon_lat'], user_id, video_id)
                print(file_name)
                data = pandas.read_csv(file_name)
                data = np.array(data)
                data = data[:,col_id] # 取经度或纬度数据

                train_size = int(len(data) * validate_percent)
                train = data[:train_size]
                validate = data[train_size:]
                x_train, y_train = split_sequence(train, history_step, predict_step)

                x_train_all.extend(x_train)
                y_train_all.extend(y_train)

            x_train_all = np.array(x_train_all)
            y_train_all = np.array(y_train_all)
            model.fit(x_train_all, y_train_all)

            # 每个用户 in 20%
            for user_id in range(user_train_size + 1, user_num + 1):
                file_name = '{}/{}/video_{}.csv'.format(config['data_dir_lon_lat'], user_id, video_id)
                data = pandas.read_csv(file_name)
                data = np.array(data)
                data = data[:,col_id] # 取经度或纬度数据

                x_test, y_test = split_sequence(data, history_step, predict_step)
                y_predict = model.predict(x_test)

                output = np.vstack((y_test, y_predict))
                output_path = './output/video_{}/{}/{}_h{}p{}_{}.csv'.format(video_id, user_id, model_name, history_step, predict_step, col_name)
                save_data_without_header(output, output_path)


