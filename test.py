import json
import pandas
import matplotlib.pyplot as plt

config = json.load(open('./config.json', 'r'))
print(config)

for uid in range(1, 11):
    data = pandas.read_csv('./data/data_lon_lat/' + str(uid) + '/video_3.csv', usecols=[1])
    plt.plot(data)
plt.show()

# data = pandas.read_csv('./data/data_lon_lat/10/video_3.csv', usecols=[1])
# plt.plot(data)
# plt.show()