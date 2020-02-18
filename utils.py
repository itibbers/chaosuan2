import os
import csv
import errno
import numpy as np
# from sklearn.preprocessing import MinMaxScaler

def normalize_data(data, scaler, feature_len):
    minmaxscaler = scaler.fit(data)
    normalize_data = minmaxscaler.transform(data)
    return normalize_data

def inverse_normalize_data(data, scaler):
    return scaler.inverse_transform(data)

def load_data(data, time_step=20, predict_step=1, validate_percent=0.67):
    seq_length = time_step + predict_step
    result = []
    for index in range(len(data) - seq_length + 1):
        result.append(data[index: index + seq_length])

    result = np.array(result)
    print('total data: ', result.shape)

    train_size = int(len(result) * validate_percent)
    train = result[:train_size, :]
    validate = result[train_size:, :]

    x_train = train[:, :time_step]
    y_train = train[:, time_step:]
    x_validate = validate[:, :time_step]
    y_validate = validate[:, time_step:]

    return [x_train, y_train, x_validate, y_validate]

# 将数据划分为每25一组 n*25*3
def divide_data(data, time_step=20, predict_step=1):
    seq_length = time_step + predict_step
    result = []
    num = int(len(data) / seq_length)
    for index in range(num):
        result.append(data[index * seq_length: (index + 1) * seq_length])

    result = np.array(result)
    print('total data: ', result.shape)

    x_train = result[:, :time_step]
    y_train = result[:, time_step:]

    return [x_train, y_train]

def save_data(data, file_path):
    print('Saving:', file_path)
    # file_path = 'outputs/output_{}.csv'.format(model_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(file_path, 'w+') as file:
        w = csv.writer(file)

        w.writerow(['x','y','z'])
        w.writerows(data)

def save_model(model, model_name):
    file_path = 'model/{}.h5'.format(model_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    model.save(file_path)

# for mlp
def split_sequence(sequence, time_step, predict_step):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + time_step
        out_end_ix = end_ix + predict_step
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# for mlp
def save_data_without_header(data, file_path):
    print('Saving:', file_path)
    # file_path = 'outputs/output_{}.csv'.format(model_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(file_path, 'w+') as file:
        w = csv.writer(file)
        w.writerows(data)

def cal_diff(data): #n*1
    diff_lon = []
    for i in range(len(data)-1):
        lon1 = data[i]
        lon2 = data[i+1]
        delta = (lon2-lon1+180)%360-180
        diff_lon.append(delta)

    return np.array(diff_lon)
