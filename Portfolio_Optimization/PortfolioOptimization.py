import os
import numpy as np # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web
from pandas_datareader import data, wb
import scipy.optimize as sco
from scipy import stats
from pandas.testing import assert_frame_equal



import math
import seaborn as sns
import datetime as dt
from datetime import datetime
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from pylab import rcParams
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D","#93D30C","#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 18, 10
RANDOM_SEED = 42 # I'm seeding with 42, if you try the same you will get the same results.
np.random.seed(RANDOM_SEED)


import math
import pandas as pd
from yahoofinancials import YahooFinancials
import yfinance as yf
GOOG = yf.download("GOOG", start="2012-05-18", end="2023-01-01",group_by="ticker") # Stock of Google
AAPL = yf.download("AAPL", start="2012-05-18", end="2023-01-01",group_by="ticker") # Stock of Apple
META = yf.download("META", start="2012-05-18", end="2023-01-01",group_by="ticker") # Stock of Facebook
AMZN = yf.download("AMZN", start="2012-05-18", end="2023-01-01",group_by="ticker") # Stock of Amazon
MSFT = yf.download("MSFT", start="2012-05-18", end="2023-01-01",group_by="ticker") # Stock of Microsoft
GSPC = yf.download("^GSPC", start="2012-05-18", end="2023-01-01",group_by="ticker") # Stock of S&P 500
# print(GOOG.shape, AAPL.shape, META.shape, AMZN.shape,MSFT.shape,GSPC.shape)

GOOG_close = GOOG[('GOOG', 'Close')]
AAPL_close = AAPL[('AAPL', 'Close')]
META_close = META[('META', 'Close')]
AMZN_close = AMZN[('AMZN', 'Close')]
MSFT_close = MSFT[('MSFT', 'Close')]
GSPC_close = GSPC[('^GSPC', 'Close')]

# Concaténation - notez que nous passons une liste à pd.concat / Axis = 1 pour concaténer les colonnes
dataset = pd.concat([
    GOOG_close,
    AAPL_close,
    META_close,
    AMZN_close,
    MSFT_close,
    GSPC_close
], axis=1)

# print(GOOG.columns)

# dataset = pd.concat([GOOG.Close, AAPL.Close, META.Close, AMZN.Close, MSFT.Close, GSPC.Close], axis=1)
# Expression above doesn't work because of multiindexing

# Renommer les colonnes
dataset.columns = ['GOOG', 'AAPL', 'META', 'AMZN', 'MSFT', 'GSPC']

# print(dataset.head())
# plt.style.use("fivethirtyeight")
# dataset[['GOOG','AAPL','META','AMZN','MSFT']].boxplot()
# plt.title("Boxplot of Stock Prices (Google, Apple, Facebook, Amazon, Microsoft)")
# # plt.show()

# pd.plotting.scatter_matrix(dataset[['GOOG','AAPL','META','AMZN','MSFT']], figsize=(10,10))
# plt.show()


plt.figure(figsize=(20,8)) # Increases the Plot Size
plt.grid(True)
plt.title('Daily Close Prices of GAFAM')
plt.xlabel('Date: May 18th, 2012 - Dec. 30th, 2022')
plt.ylabel('Values')
plt.plot(dataset['GOOG'], 'red', label='Google')
plt.plot(dataset['AAPL'], 'black', label='Apple')
plt.plot(dataset['META'], 'blue', label='Facebook')
plt.plot(dataset['AMZN'], 'orange', label='Amazon')
plt.plot(dataset['MSFT'], 'green', label='Microsoft')
plt.legend()
# plt.show()

dataset['R_GOOG'] = dataset[['GOOG']].pct_change(1)
dataset['R_AAPL'] = dataset[['AAPL']].pct_change(1)
dataset['R_META'] = dataset[['META']].pct_change(1)
dataset['R_AMZN'] = dataset[['AMZN']].pct_change(1)
dataset['R_MSFT'] = dataset[['MSFT']].pct_change(1)
dataset['R_GSPC'] = dataset[['GSPC']].pct_change(1)
print(dataset.head())
print(dataset.describe())

with sns.axes_style("whitegrid"):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,12))

    axes[0][0].plot(dataset['R_GOOG'], color='red')
    axes[0][0].set_title('Daily Returns  of Google ')

    sns.distplot(dataset['R_GOOG'], norm_hist=True, fit=stats.norm, color='red',
                bins=50, ax=axes[0][1])
    axes[0][1].set_title('Density of Google daily returns')

    axes[1][0].plot(dataset['R_AAPL'], color='black')
    axes[1][0].set_title('Daily Returns  of Apple')

    sns.distplot(dataset['R_AAPL'], norm_hist=True, fit=stats.norm, color='black',
                bins=50, ax=axes[1][1])
    axes[1][1].set_title('Density of Apple daily returns')
    plt.tight_layout()
    plt.show()

mode = stats.mode(dataset['R_AAPL'])

# Médiane (égale à describe().loc['50%'])
mediane = dataset['R_AAPL'].median()

print(f"\nMédiane : {mediane}")
print(f"Mode : {mode}")