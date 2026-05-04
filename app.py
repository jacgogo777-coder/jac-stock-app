import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance.original_flavor as mpf
import pandas as pd
import numpy as np
import matplotlib

# ==========================================
# 0. 網頁基本設定與標題
# ==========================================
st.set_page_config(page_title="台股技術分析儀表板", layout="wide")

# 設定中文字型 (確保 Matplotlib 繪圖能顯示中文)
try:
    matplotlib.rc('font', family='Microsoft JhengHei')
except:
    pass # 若在 Linux/雲端環境無正黑體，則使用預設字體
matplotlib.rc('axes', unicode_minus=False)

st.title("📈 課程專題：台股技術分析視覺化儀表板")
st.markdown("""
本專題將 Python 股市分析腳本轉換為 **Streamlit 互動式網頁**。
透過左側邊欄設定參數，系統會自動下載資料、計算各項技術指標 (均線、布林帶、KDJ、MACD、OBV)，並即時繪製專業綜合 K 線圖。
""")

# ==========================================
# 1. 分段說明：側邊欄與參數設定
# ==========================================
st.sidebar.header("⚙️ 參數設定")
st.sidebar.markdown("請設定您想查詢的股票代號與日期區間：")

stock_id = st.sidebar.text_input("股票代號 (如: 2330.TW)", "2330.TW")
start_date = st.sidebar.date_input("開始日期", datetime(2024, 10, 18))
end_date = st.sidebar.date_input("結束日期", datetime(2025, 4, 20))

# ==========================================
# 2. 分段說明：資料獲取與前處理
# ==========================================
st.header("步驟 1：獲取股市資料與前處理")
with st.expander("📖 查看本段程式設計說明"):
    st.markdown("""
    - 使用 `yfinance` 庫動態下載股市資料。
    - 處理回傳的多層級索引 (`MultiIndex`) 問題，將其展平。
    - 將日期索引轉換為 `%y-%m-%d` 的字串格式，方便後續 X 軸繪圖標示。
    """)

# 使用快取機制，避免每次互動都重新下載資料
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if df.empty:
        return df
    
    # 日期格式化
    df.index = df.index.map(lambda x: x.strftime('%y-%m-%d'))
    
    # 展平欄位名稱 (若 yfinance 回傳 MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

df = load_data(stock_id, start_date, end_date)

if df.empty:
    st.error("找不到該股票資料或無此交易區間，請確認代號或日期。")
    st.stop()

st.success(f"✅ 成功獲取 **{stock_id}** 資料，共 {len(df)} 筆交易紀錄。")
st.dataframe(df.tail()) # 顯示最後五筆資料預覽

# ==========================================
# 3. 分段說明：技術指標計算
# ==========================================
st.header("步驟 2：技術指標計算 (SMA, BBands, KDJ, OBV, MACD)")
with st.expander("📖 查看指標運算邏輯說明"):
    st.markdown("""
    - **SMA (簡單移動平均線)**：計算 5日、20日、60日均線。
    - **布林通道 (Bollinger Bands)**：以 20日均線為中軌，上下加減 2 倍標準差。
    - **KDJ 指標**：先求取 9 日的 RSV 值，再利用平滑公式依序計算 K、D、J 值。
    - **OBV (能量潮指標)**：以當日收盤價與前一日比較，上漲則加上成交量，下跌則扣除。
    - **MACD (指數平滑異同移動平均線)**：計算 12日與 26日的 EMA，得出 DIF 與 MACD，並計算柱狀圖 (Histogram)。
    """)

with st.spinner('指標計算中...'):
    # 複製一份資料進行計算
    df = df.copy()
    
    # --- 計算SMA ---
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()

    # --- 計算布林帶 ---
    df['middle_band'] = df['SMA_20']
    df['std_dev'] = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)

    # --- 計算KDJ線 ---
    n = 9
    low_min = df['Low'].rolling(window=n).min()
    high_max = df['High'].rolling(window=n).max()
    df['RSV'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
    
    df['K'] = pd.Series(index=df.index, dtype=float)
    df['D'] = pd.Series(index=df.index, dtype=float)
    
    if len(df) > 9:
        df.loc[df.index[8], 'K'] = 50
        df.loc[df.index[8], 'D'] = 50
        for i in range(9, len(df)):
            df.loc[df.index[i], 'K'] = df.loc[df.index[i-1], 'K'] * (2/3) + df.loc[df.index[i], 'RSV'] * (1/3)
            df.loc[df.index[i], 'D'] = df.loc[df.index[i-1], 'D'] * (2/3) + df.loc[df.index[i], 'K'] * (1/3)
            
    df['3K-2D'] = 3 * df['K'] - 2 * df['D']
    df['J'] = 3 * df['D'] - 2 * df['K']

    # --- 計算OBV ---
    df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], -df['Volume'])
    df['OBV'] = df['OBV'].cumsum()

    # --- 計算 MACD ---
    fast_period = 12
    slow_period = 26
    signal_period = 9
    
    # 避免索引超出範圍的保護
    if len(df) > slow_period:
        df.loc[df.index[12]:,'EMA12'] = df['Adj Close'].ewm(span=12, adjust=False).mean()
        df.loc[df.index[26]:,'EMA26'] = df['Adj Close'].ewm(span=26, adjust=False).mean()
        df.loc[:df.index[fast_period-1], 'EMA12'] = 0
        df.loc[:df.index[slow_period-1], 'EMA26'] = 0
        
        df['DIF'] = df['EMA12'] - df['EMA26']
        df.loc[:df.index[slow_period-1], 'DIF'] = 0
        df['MACD'] = df['DIF'].ewm(span=signal_period, adjust=False).mean()
        df['MACD Histogram'] = df['DIF'] - df['MACD']
    else:
        # 資料太少時填補 0
        df['DIF'] = 0
        df['MACD'] = 0
        df['MACD Histogram'] = 0

    macd_colors = np.where(df['MACD Histogram'] >= 0, 'r', 'g')

# ==========================================
# 4. 分段說明：視覺化繪圖
# ==========================================
st.header("步驟 3：綜合指標視覺化圖表")
with st.expander("📖 查看圖表排版說明"):
    st.markdown("""
    這是一個高度客製化的 Matplotlib 圖表，使用 `add_subplot` 切割為 6 等分：
    1. **主圖 (上 1~3 等分)**：包含紅綠 K線圖 (`candlestick2_ochl`)、5/20/60日均線以及紫色虛線的布林通道。
    2. **KDJ 圖 (第 4 等分)**：青色K線、紫色D線、橘色虛線J線的黃金/死亡交叉觀察。
    3. **OBV 與 成交量圖 (第 5 等分)**：左軸為 OBV 趨勢，右軸(`twinx`)繪製紅綠成交量柱狀圖。
    4. **MACD 圖 (第 6 等分)**：繪製 DIF、MACD 線，並以紅綠色區分 MACD 柱狀圖。
    """)

# 建立畫布
fig = plt.figure(figsize=(14, 10), layout='constrained')

# --- 繪製 K 線圖與均線 (佔比 1-3) ---
ax1 = fig.add_subplot(6, 1, (1,3))
ax1.set_xticks(range(0, len(df.index), 10))
ax1.set_xticklabels(df.index[::10])
mpf.candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'], df['Low'], width=0.8, colorup='r', colordown='g', alpha=1)
ax1.plot(df['SMA_5'], label='5日均線', alpha=0.9, color='b')
ax1.plot(df['SMA_20'], label='20日均線', alpha=0.9, color='orange')
ax1.plot(df['SMA_60'], label='60日均線', alpha=0.9, color='g')
ax1.plot(df['upper_band'], label='upperband', alpha=0.9, color='purple', ls=':')
ax1.plot(df['lower_band'], label='lowerband', alpha=0.9, color='purple', ls=':')
ax1.legend(loc=0)
# 動態設定標題
ax1.set_title(f"【{stock_id}】綜合技術指標分析 (K線/均線/布林帶/KDJ/OBV/MACD)", fontsize=16)

# --- 繪製 KDJ (佔比 4) ---
ax3 = fig.add_subplot(6, 1, 4)
ax3.plot(df['K'], label='K line', color='cyan')
ax3.plot(df['D'], label='D line', color='purple')
ax3.plot(df['J'], label='J line', linestyle='--', color='orange')
ax3.set_xticks(range(0, len(df.index), 10))
ax3.set_xticklabels([])
ax3.legend(loc=0)

# --- 繪製 OBV 與 交易量 (佔比 5) ---
ax2 = fig.add_subplot(6, 1, 5)
ax2.set_xticks(range(0, len(df.index), 10))
ax2.set_xticklabels([])

# 自訂漲跌趨勢交易，前二天預設為灰色
colors = np.where(df['Close'] > df['Close'].shift(1), 'r', 'g')
colors = np.array(['gray', 'gray'] + list(colors[2:]))

ax2.plot(df['OBV'], color='purple', linestyle='--', label='OBV')
ax2.legend(loc=2)

ax2_1 = ax2.twinx()
ax2_1.bar(df.index, height=df['Volume'], color=colors, width=0.8, alpha=0.8, label='Volume')
ax2_1.legend(loc=1)

# --- 繪製 MACD (佔比 6) ---
ax4 = fig.add_subplot(6, 1, 6)
ax4.plot(df['DIF'], label='DIF', color='orange')
ax4.plot(df['MACD'], label='MACD', color='blue')
ax4.bar(df.index, height=df['MACD Histogram'], color=macd_colors, label='MACD Histogram')
ax4.set_xticks(range(0, len(df.index), 10))
ax4.set_xticklabels(df.index[::10], rotation=45)
ax4.legend(loc=2)

# ==========================================
# 5. 網頁渲染圖表
# ==========================================
# 將 Matplotlib 的 Figure 傳遞給 Streamlit 渲染
st.pyplot(fig)

st.divider()
st.caption("🚀 本頁面使用 Python Streamlit + Matplotlib 打造 | 資料來源: Yahoo Finance")