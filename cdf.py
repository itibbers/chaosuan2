import pandas
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

data = pandas.read_csv('./video1-lat.csv', header=None)
data = np.array(data)

bar_width = 0.2
x = np.linspace(1,5,5)
for i in range(4):
  plt.bar(x+i*bar_width, data[i], align='center', width=bar_width)
  plt.xticks(x + bar_width*1.5, list(map(lambda x:int(x), x)))
plt.show()