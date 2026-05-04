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

# ==========================================
# 0. 網頁基本配置與字型處理
# ==========================================
st.set_page_config(page_title="台股技術分析教學儀表板", layout="wide")

# --- 強化版中文字型處理 (解決雲端與本地亂碼問題) ---
font_path = "NotoSansTC-Regular.ttf"

@st.cache_resource
def init_font(path):
    """
    載入並註冊中文字型，確保圖表標題與標籤能正確顯示中文
    """
    if os.path.exists(path):
        try:
            # 註冊字型到 Matplotlib
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            font_name = prop.get_name()
            # 設定全域字型偏好
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False # 解決負號變方塊問題
            return font_name
        except Exception as e:
            return f"Error: {e}"
    return None

# 初始化字型
target_font_name = init_font(font_path)

# 標題與簡介
st.title("📈 課程專題：台股技術分析視覺化儀表板")
st.markdown("""
本專案將 Python 股市分析腳本轉換為 **Streamlit 互動式網頁**。
您可以透過左側選單動態調整參數，系統將自動計算並繪製包含 **均線、布林帶、KDJ、OBV、MACD** 的綜合技術分析圖表。
""")

# ==========================================
# 1. 側邊欄與參數設定
# ==========================================
st.sidebar.header("⚙️ 參數設定")
st.sidebar.markdown("請設定您想查詢的股票代號與日期區間：")

stock_id = st.sidebar.text_input("股票代號 (如: 2330.TW)", "2330.TW")
start_date = st.sidebar.date_input("開始日期", datetime(2025, 10, 10))
end_date = st.sidebar.date_input("結束日期", datetime(2026, 5, 5))

# 顯示字型狀態診斷
if target_font_name and not target_font_name.startswith("Error"):
    st.sidebar.success(f"✅ 已載入中文字型: {target_font_name}")
else:
    st.sidebar.error("❌ 找不到 NotoSansTC-Regular.ttf")
    st.sidebar.info("部署至雲端時，請確保 GitHub 根目錄有該字型檔。")

# ==========================================
# 2. 步驟 1：獲取資料 (Step 1)
# ==========================================
st.header("步驟 1：獲取股市資料與前處理")
with st.expander("📖 查看本段程式設計說明"):
    st.markdown("""
    - 使用 `yfinance` 庫從 Yahoo Finance 下載原始交易資料。
    - **資料清洗**：處理下載後可能產生的多層級索引 (`MultiIndex`)。
    - **格式化日期**：將索引轉為 `%y-%m-%d` 字串，避免繪圖時 X 軸時間軸間距不均。
    - **快取機制**：使用 `@st.cache_data` 減少重複下載，提升網頁反應速度。
    """)

@st.cache_data
def get_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, auto_adjust=False)
    if not data.empty:
        # 日期格式化為字串
        data.index = data.index.map(lambda x: x.strftime('%y-%m-%d'))
        # 展平欄位名稱 (若 yfinance 回傳 MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    return data

df = get_data(stock_id, start_date, end_date)

if df.empty:
    st.error("找不到資料，請檢查代號或日期設定。")
    st.stop()

st.success(f"✅ 成功獲取 **{stock_id}** 資料，最後五筆預覽：")
st.dataframe(df.tail())

# ==========================================
# 3. 步驟 2：技術指標計算 (Step 2)
# ==========================================
st.header("步驟 2：技術指標計算")
with st.expander("📖 查看指標運算邏輯說明"):
    st.markdown("""
    - **均線 (SMA)**：5日、20日、60日移動平均線，觀察長短期趨勢。
    - **布林通道 (BBands)**：以20日均線為中心，計算上下 2 倍標準差。
    - **KDJ 指標**：計算 9 日 RSV 並透過權重平滑計算 K、D、J 值。
    - **OBV (能量潮)**：利用成交量與股價漲跌的累計值，觀察資金動向。
    - **MACD**：計算 12/26 日 EMA 之差 (DIF) 及其 9 日平均 (MACD)。
    """)

with st.spinner('各項指標計算中...'):
    df = df.copy()
    # 均線
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    # 布林帶
    df['std_dev'] = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['SMA_20'] + (df['std_dev'] * 2)
    df['lower_band'] = df['SMA_20'] - (df['std_dev'] * 2)
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

# ==========================================
# 4. 步驟 3：視覺化圖表 (Step 3)
# ==========================================
st.header("步驟 3：綜合指標視覺化圖表")
with st.expander("📖 查看圖表排版設計說明"):
    st.markdown("""
    本圖表使用 Matplotlib 的 `add_subplot` 將畫布切分為 6 個區塊：
    - **區塊 1-3 (主圖)**：顯示 K 線、三條均線與紫色虛線的布林通道。
    - **區塊 4 (KDJ)**：觀察 K、D、J 三線的交叉與數值。
    - **區塊 5 (OBV & Volume)**：同時顯示能量潮曲線與對應收盤漲跌顏色的成交量柱狀圖。
    - **區塊 6 (MACD)**：顯示 DIF、MACD 指標及其紅綠柱狀圖。
    - **視覺優化**：中間圖表的 X 軸標籤已隱藏，避免文字疊加，僅在最底部顯示日期。
    """)

# 設定標題字體 (若註冊成功則指定，否則留空使用全域設定)
title_font = {'fontname': target_font_name} if target_font_name else {}

# 建立圖表畫布
fig = plt.figure(figsize=(14, 12), layout='constrained')

# --- 1. 主圖：K線圖與均線 (佔比 1-3) ---
ax1 = fig.add_subplot(6, 1, (1,3))
ax1.set_xticks(range(0, len(df.index), 10))
ax1.set_xticklabels(df.index[::10])
mpf.candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'], df['Low'], 
                       width=0.8, colorup='r', colordown='g', alpha=1)
ax1.plot(df['SMA_5'], label='5日均線', alpha=0.9, color='b')
ax1.plot(df['SMA_20'], label='20日均線', alpha=0.9, color='orange')
ax1.plot(df['SMA_60'], label='60日均線', alpha=0.9, color='g')
ax1.plot(df['upper_band'], label='布林上軌', color='purple', ls=':', alpha=0.8)
ax1.plot(df['lower_band'], label='布林下軌', color='purple', ls=':', alpha=0.8)
ax1.set_title(f"【{stock_id}】技術分析綜合圖表", fontsize=16, **title_font)
ax1.legend(loc='upper left')

# --- 2. KDJ 指標圖 (佔比 4) ---
ax3 = fig.add_subplot(6, 1, 4)
ax3.plot(df['K'], label='K線', color='cyan')
ax3.plot(df['D'], label='D線', color='purple')
ax3.plot(df['J'], label='J線', linestyle='--', color='orange')
ax3.set_xticks(range(0, len(df.index), 10))
ax3.set_xticklabels([]) # 隱藏 X 軸刻度避免重疊
ax3.set_title("KDJ 指標分析", **title_font)
ax3.legend(loc='upper left')

# --- 3. OBV 與 成交量圖 (佔比 5) ---
ax2 = fig.add_subplot(6, 1, 5)
ax2.set_xticks(range(0, len(df.index), 10))
ax2.set_xticklabels([]) # 隱藏 X 軸刻度
ax2.plot(df['OBV'], color='purple', linestyle='--', label='OBV')
ax2.set_title("OBV 能量潮與成交量趨勢", **title_font)
ax2.legend(loc='upper left')

ax2_1 = ax2.twinx()
vol_colors = np.where(df['Close'] > df['Close'].shift(1), 'r', 'g')
ax2_1.bar(df.index, df['Volume'], color=vol_colors, width=0.8, alpha=0.4, label='成交量')

# --- 4. MACD 指標圖 (佔比 6) ---
ax4 = fig.add_subplot(6, 1, 6)
m_colors = np.where(df['MACD Histogram'] >= 0, 'r', 'g')
ax4.plot(df['DIF'], label='DIF (快線)', color='orange')
ax4.plot(df['MACD'], label='MACD (慢線)', color='blue')
ax4.bar(df.index, df['MACD Histogram'], color=m_colors, label='MACD 柱狀圖')
ax4.set_xticks(range(0, len(df.index), 10))
ax4.set_xticklabels(df.index[::10], rotation=45)
ax4.set_title("MACD 指標分析", **title_font)
ax4.legend(loc='upper left')

# 網頁渲染圖表
st.pyplot(fig)

st.divider()
st.caption("🚀 專題實作：Python 資料視覺化課程 | 資料來源: Yahoo Finance")