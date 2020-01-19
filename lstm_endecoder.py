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

def Model(time_step=30, predict_step=15, feature_len=1):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(time_step, feature_len)))
    model.add(RepeatVector(predict_step))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    return model


model_name = sys.argv[0].replace(".py", "")
config = json.load(open('./config.json'))

video_id = config['video_id']
col_class = config['col_class']
user_num = config['user_num']
validate_percent = config['validate_percent']
user_train_size = int(user_num * validate_percent)
history_class = config['history_class']
predict_class = config['predict_class']

scaler = MinMaxScaler(feature_range=(0, 1))
feature_len = 1

for col_id in range(len(col_class)):
    col_name = col_class[col_id]

    for history_id in range(len(history_class)):
        history_step = history_class[history_id]

        for predict_id in range(len(predict_class)):
            predict_step = predict_class[predict_id]
            model_class_name = 'video_{}_{}_h{}p{}_{}'.format(video_id, model_name, history_step, predict_step, col_name)
            model = Model(history_step, predict_step)

            x_train_all = []
            y_train_all = []
            x_validate_all = []
            y_validate_all = []
            # 每个用户 in 80%
            for user_id in range(1, user_train_size + 1):
                file_name = '{}/{}/video_{}.csv'.format(config['data_dir_lon_lat'], user_id, video_id)
                print(file_name)
                data = pandas.read_csv(file_name, usecols=[col_id])
                data = np.array(data)
                data = normalize_data(data, scaler, feature_len)

                train_size = int(len(data) * validate_percent)
                train = data[:train_size]
                validate = data[train_size:]
                x_train, y_train = split_sequence(train, history_step, predict_step)
                x_validate, y_validate = split_sequence(validate, history_step, predict_step)

                x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
                x_validate = x_validate.reshape(x_validate.shape[0], x_validate.shape[1], 1)
                y_validate = y_validate.reshape(y_validate.shape[0], y_validate.shape[1], 1)

                x_train_all.extend(x_train)
                y_train_all.extend(y_train)
                x_validate_all.extend(x_validate)
                y_validate_all.extend(y_validate)

            x_train_all = np.array(x_train_all)
            y_train_all = np.array(y_train_all)
            x_validate_all = np.array(x_validate_all)
            y_validate_all = np.array(y_validate_all)
            model.fit(x_train_all, y_train_all, batch_size=config['batch_size'], epochs=config['epochs'], validation_data=(x_validate_all, y_validate_all))

            # 每个用户 in 20%
            for user_id in range(user_train_size + 1, user_num + 1):
                file_name = '{}/{}/video_{}.csv'.format(config['data_dir_lon_lat'], user_id, video_id)
                data = pandas.read_csv(file_name, usecols=[col_id])
                data = np.array(data)
                data = normalize_data(data, scaler, feature_len)

                x_test, y_test = split_sequence(data, history_step, predict_step)
                x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

                y_predict = model.predict(x_test)

                y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
                y_predict = y_predict.reshape(y_test.shape[0], y_test.shape[1])

                y_test = inverse_normalize_data(y_test, scaler)
                y_predict = inverse_normalize_data(y_predict, scaler)

                output = np.vstack((y_test, y_predict))
                output_path = './output/video_{}/{}/{}_h{}p{}_{}.csv'.format(video_id, user_id, model_name, history_step, predict_step, col_name)
                save_data_without_header(output, output_path)

