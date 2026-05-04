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

# --- 1. 網頁基本配置 ---
st.set_page_config(page_title="2026 股市實作專案", layout="wide")

# --- 2. 強化版中文字型處理 (解決雲端亂碼) ---
font_path = "NotoSansTC-Regular.ttf"

@st.cache_resource
def init_font(path):
    if os.path.exists(path):
        try:
            # 註冊字型到 Matplotlib
            font_manager.fontManager.addfont(path)
            # 取得字型名稱
            prop = font_manager.FontProperties(fname=path)
            font_name = prop.get_name()
            # 設定全域字型
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            return font_name
        except Exception as e:
            return f"Error: {e}"
    return None

target_font_name = init_font(font_path)

# 在側邊欄顯示狀態診斷
if target_font_name and not target_font_name.startswith("Error"):
    st.sidebar.success(f"✅ 已載入中文字型: {target_font_name}")
else:
    st.sidebar.error("❌ 找不到 NotoSansTC-Regular.ttf")
    st.sidebar.info("請確認 GitHub 根目錄是否有該檔案，且檔名完全正確。")

st.title("📈 2026 歡慶端午 2330 股市分析實作")

# --- 3. 側邊欄參數 ---
st.sidebar.header("查詢參數")
stock_id = st.sidebar.text_input("股票代號", "2330.TW")
start_date = st.sidebar.date_input("開始日期", datetime(2024, 10, 18))
end_date = st.sidebar.date_input("結束日期", datetime(2025, 4, 20))

# --- 4. 資料處理 ---
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
    st.error("找不到資料。")
    st.stop()

# --- 5. 指標計算 (保持您的原始邏輯) ---
df = df.copy()
# 均線與布林帶
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_60'] = df['Close'].rolling(window=60).mean()
df['upper_band'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
df['lower_band'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
# KDJ
n = 9
low_min = df['Low'].rolling(window=n).min()
high_max = df['High'].rolling(window=n).max()
df['RSV'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
df['K'] = pd.Series(50.0, index=df.index)
df['D'] = pd.Series(50.0, index=df.index)
for i in range(9, len(df)):
    df.loc[df.index[i], 'K'] = df.loc[df.index[i-1], 'K'] * (2/3) + df.loc[df.index[i], 'RSV'] * (1/3)
    df.loc[df.index[i], 'D'] = df.loc[df.index[i-1], 'D'] * (2/3) + df.loc[df.index[i], 'K'] * (1/3)
df['J'] = 3 * df['D'] - 2 * df['K']
# OBV
df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], -df['Volume']).cumsum()
# MACD
ema12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
df['DIF'] = ema12 - ema26
df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()
df['MACD Histogram'] = df['DIF'] - df['MACD']

# --- 6. 繪圖與視覺化 ---
# 強制指定中文字型物件以確保渲染
title_font = {'fontname': target_font_name} if target_font_name else {}

fig = plt.figure(figsize=(14, 12), layout='constrained')

# 主圖
ax1 = fig.add_subplot(6,1,(1,3))
ax1.set_xticks(range(0,len(df.index),10))
ax1.set_xticklabels(df.index[::10])
mpf.candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'], df['Low'], width=0.8, colorup='r', colordown='g')
ax1.plot(df['SMA_5'], label='5日均線')
ax1.plot(df['SMA_20'], label='20日均線')
ax1.plot(df['SMA_60'], label='60日均線')
ax1.plot(df['upper_band'], color='purple', ls=':', label='布林上軌')
ax1.plot(df['lower_band'], color='purple', ls=':', label='布林下軌')
ax1.set_title(f"【{stock_id}】技術分析綜合圖表", fontsize=16, **title_font)
ax1.legend(loc='upper left')

# KDJ
ax3 = fig.add_subplot(6,1,4)
ax3.plot(df['K'], label='K線')
ax3.plot(df['D'], label='D線')
ax3.plot(df['J'], label='J線', ls='--')
ax3.set_title("KDJ 指標分析", **title_font)
ax3.legend(loc='upper left')

# OBV
ax2 = fig.add_subplot(6,1,5)
ax2.plot(df['OBV'], color='purple', ls='--', label='OBV')
ax2.set_title("OBV 能量潮趨勢", **title_font)
ax2_1 = ax2.twinx()
vol_colors = np.where(df['Close'] > df['Close'].shift(1), 'r', 'g')
ax2_1.bar(df.index, df['Volume'], color=vol_colors, alpha=0.3)

# MACD
ax4 = fig.add_subplot(6,1,6)
m_colors = np.where(df['MACD Histogram'] >= 0, 'r', 'g')
ax4.plot(df['DIF'], label='DIF', color='orange')
ax4.plot(df['MACD'], label='MACD', color='blue')
ax4.bar(df.index, df['MACD Histogram'], color=m_colors)
ax4.set_xticks(range(0,len(df.index),10))
ax4.set_xticklabels(df.index[::10], rotation=45)
ax4.set_title("MACD 指標分析", **title_font)

st.pyplot(fig)