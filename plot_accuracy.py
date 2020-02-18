import pandas
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import *
import json

config = json.load(open('./config.json'))
history_class = config['history_class']
predict_class = config['predict_class']
model_class = ['lstm_m2o', 'lr2', 'mlp_seq', 'naive']
model_class_dummy = ['naive', 'lr2', 'mlp_seq', 'lstm_m2o']
model_len = len(model_class)
history_len = len(history_class)
predict_len = len(predict_class)
video_id = config['video_id']
col_name = config['col_class'][0]

# 数据目录
data_dir = './output/'

# 准确度
accuracy = 20

# 柱状图在横坐标上的位置
x = np.arange(predict_len)
# 设置柱状图的宽度
bar_width = 0.15

# 遍历参数
for i in range(history_len):
    # 遍历模型
    for model_id in range(model_len):
        model_name = model_class_dummy[model_id]
        y = []
        # 遍历参数
        for j in range(predict_len):
            history_step = history_class[i]
            predict_step = predict_class[j]
            params_name = 'h{}p{}_{}'.format(history_step, predict_step, col_name)

            file_name = '{}video_{}/cdf/csv/{}_{}.csv'.format(data_dir, video_id, model_name, params_name)
            print(file_name)
            data = pandas.read_csv(file_name, header=None)
            data = np.array(data)

            num = 0
            for v in range(len(data)):
                if data[v][0] <= accuracy:
                    num = num + 1
            acc = num / len(data)
            y.append(acc)
        plt.bar(x + model_id * bar_width, y, bar_width, label=model_class[model_id])

    plt.xlabel('时间间隔(s)')
    plt.ylabel('准确度(%)')
    plt.legend(loc='upper right')
    # 显示x坐标轴的标签,即tick_label=model_class,调整位置，使其落在两个直方图中间位置
    plt.xticks(x + bar_width * (len(model_class) - 1) / 2, list(map(lambda x:x//30, predict_class)))
    file_path = '{}video_{}/accuracy/a{}h{}_{}.png'.format(data_dir, video_id, accuracy, history_class[i], col_name)
    plt.savefig(file_path, dpi=200)
    plt.show()
