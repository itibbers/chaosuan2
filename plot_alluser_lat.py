import pandas
import numpy as np
import math
import matplotlib.pyplot as plt
import json

config = json.load(open('./config.json'))

for i in range(1, config['user_num'] + 1):
    data = pandas.read_csv(config['data_dir_lon_lat'] + '/' + str(i) + '/video_1.csv', usecols=[0])
    data = np.array(data)
    plt.plot(data)

plt.show()
