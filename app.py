import streamlit as st
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import mplfinance.original_flavor as mpf
import pandas as pd
import numpy as np
import matplotlib
import os
from matplotlib import font_manager

# --- 網頁配置 ---
st.set_page_config(page_title="2026 股市實作專案", layout="wide")

# --- 中文字型處理 ---
# 尋找專案目錄下的字型檔 (例如 NotoSansTC-Regular.ttf 或 msjh.ttf)
font_filename = "NotoSansTC-Regular.ttf" # 建議下載此字型放進專案
if os.path.exists(font_filename):
    font_ptr = font_manager.FontProperties(fname=font_filename)
    matplotlib.rc('font', family=font_ptr.get_name())
    # 解決負號顯示問題
    plt.rcParams['axes.unicode_minus'] = False
else:
    # 如果沒字型，嘗試回退到系統字型或顯示警告
    st.sidebar.warning("找不到中文字型檔，圖表標題可能顯示異常。請將字型檔上傳至 GitHub。")
    try:
        matplotlib.rc('font', family='Microsoft JhengHei')
    except:
        pass

st.title("📈 2026 歡慶端午 2330 股市分析實作")

# --- 側邊欄參數 ---
st.sidebar.header("查詢參數")
stock_id = st.sidebar.text_input("股票代號", "2330.TW")
start_date = st.sidebar.date_input("開始日期", datetime(2025, 10, 10))
end_date = st.sidebar.date_input("結束日期", datetime(2026, 5, 5))

# --- 下載資料 ---
@st.cache_data
def get_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, auto_adjust=False)
    if not data.empty:
        data.index = data.index.map(lambda x: x.strftime('%y-%m-%d'))
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    return data

df = get_data(stock_id, start_date, end_date)

if df.empty:
    st.error("找不到資料，請檢查代號或日期設定。")
    st.stop()

# --- 計算指標 (承襲原有的邏輯) ---
df = df.copy()
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_60'] = df['Close'].rolling(window=60).mean()
df['middle_band'] = df['SMA_20']
df['std_dev'] = df['Close'].rolling(window=20).std()
df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)

n = 9
low_min = df['Low'].rolling(window=n).min()
high_max = df['High'].rolling(window=n).max()
df['RSV'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
df['K'] = pd.Series(index=df.index, dtype=float)
df['D'] = pd.Series(index=df.index, dtype=float)
if len(df) > 8:
    df.loc[df.index[8], 'K'] = 50
    df.loc[df.index[8], 'D'] = 50
    for i in range(9, len(df)):
        df.loc[df.index[i], 'K'] = df.loc[df.index[i-1], 'K'] * (2/3) + df.loc[df.index[i], 'RSV'] * (1/3)
        df.loc[df.index[i], 'D'] = df.loc[df.index[i-1], 'D'] * (2/3) + df.loc[df.index[i], 'K'] * (1/3)
df['J'] = 3 * df['D'] - 2 * df['K']

df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], -df['Volume'])
df['OBV'] = df['OBV'].cumsum()

df.loc[df.index[12]:,'EMA12'] = df['Adj Close'].ewm(span=12, adjust=False).mean()
df.loc[df.index[26]:,'EMA26'] = df['Adj Close'].ewm(span=26, adjust=False).mean()
df['DIF'] = df['EMA12'] - df['EMA26']
df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()
df['MACD Histogram'] = df['DIF'] - df['MACD']
macd_colors = np.where(df['MACD Histogram'] >= 0, 'r', 'g')

# --- 繪圖 ---
fig = plt.figure(figsize=(14, 10), layout='constrained')
ax1 = fig.add_subplot(6,1,(1,3))
ax1.set_xticks(range(0,len(df.index),10))
ax1.set_xticklabels(df.index[::10])
mpf.candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'], df['Low'], width=0.8, colorup='r', colordown='g')
ax1.plot(df['SMA_5'], label='5日均線')
ax1.plot(df['SMA_20'], label='20日均線')
ax1.plot(df['SMA_60'], label='60日均線')
ax1.plot(df['upper_band'], color='purple', ls=':', label='布林上軌')
ax1.plot(df['lower_band'], color='purple', ls=':', label='布林下軌')
ax1.set_title(f"【{stock_id}】技術分析圖表")
ax1.legend(loc='upper left')

# KDJ
ax3 = fig.add_subplot(6,1,4)
ax3.plot(df['K'], label='K')
ax3.plot(df['D'], label='D')
ax3.plot(df['J'], label='J', ls='--')
ax3.set_title("KDJ 指標")
ax3.legend(loc='upper left')

# OBV
ax2 = fig.add_subplot(6,1,5)
ax2.plot(df['OBV'], color='purple', ls='--', label='OBV')
ax2.set_title("OBV 能量潮")
ax2_1 = ax2.twinx()
colors = np.where(df['Close'] > df['Close'].shift(1), 'r', 'g')
ax2_1.bar(df.index, df['Volume'], color=colors, alpha=0.3)

# MACD
ax4 = fig.add_subplot(6,1,6)
ax4.plot(df['DIF'], label='DIF', color='orange')
ax4.plot(df['MACD'], label='MACD', color='blue')
ax4.bar(df.index, df['MACD Histogram'], color=macd_colors)
ax4.set_xticks(range(0,len(df.index),10))
ax4.set_xticklabels(df.index[::10], rotation=45)
ax4.set_title("MACD 指標")

# 顯示到網頁
st.pyplot(fig)