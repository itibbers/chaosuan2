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
model_len = len(model_class)
history_len = len(history_class)
predict_len = len(predict_class)
video_id = config['video_id']

# 数据目录
data_dir = './output/'

# 准确度
accuracy = 20

# 柱状图在横坐标上的位置
x = np.arange(predict_len)
# 设置柱状图的宽度
bar_width = 0.15

lstm = [0.971, 0.964, 0.946, 0.911, 0.894, 0.885]
mlp = [0.965, 0.936, 0.903, 0.874, 0.851, 0.832]
lr = [0.969, 0.929, 0.857, 0.801, 0.763, 0.713]
naive = [0.912, 0.883, 0.813, 0.787, 0.734, 0.698]

# 遍历参数
for i in range(history_len):
    # 遍历模型
    for model_id in range(model_len):
        model_name = model_class[model_id]
        y = []
        # 遍历参数
        for j in range(predict_len):
            history_step = history_class[i]
            predict_step = predict_class[j]
            params_name = 'h' + str(history_step) + 'p' + str(predict_step) + '_lat'

            file_name = '{}cdf/csv/{}_{}.csv'.format(data_dir, model_name, params_name)
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

    plt.xlabel('predict time')
    plt.ylabel('accuracy rate')
    plt.title('准确度' + str(accuracy))
    plt.legend(loc='upper right')
    # 显示x坐标轴的标签,即tick_label=model_class,调整位置，使其落在两个直方图中间位置
    plt.xticks(x + bar_width * (len(model_class) - 1) / 2, predict_class)
    file_path = '{}accuracy/a{}h{}.png'.format(data_dir, accuracy, history_class[i])
    plt.savefig(file_path, dpi=200)
    plt.show()

