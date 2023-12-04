# To jest wersja 2 nowej strony - gdzie modele mają opisy i parametry i jest ich więcej

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import appdirs as ad
CACHE_DIR = ".cache"
# Force appdirs to say that the cache dir is .cache
ad.user_cache_dir = lambda *args: CACHE_DIR
# Create the cache dir if it doesn't exist
Path(CACHE_DIR).mkdir(exist_ok=True)
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from streamlit import set_page_config
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# Set page configuration for full width
set_page_config(layout="wide")

# Definicje
today = date.today()
comm_dict = {'^DJI':'DJI30','EURUSD=X':'USD_EUR','CNY=X':'USD/CNY',
             'CL=F':'Crude_Oil','GC=F':'Gold','^IXIC':'NASDAQ',
             '^GSPC':'SP_500','^TNX':'10_YB','AED=X':'USD/AED',
             'HG=F':'Copper','GBPUSD=X':'USD_GBP',
             'JPY=X':'USD_JPY','EURPLN=X':'EUR/PLN','PLN=X':'PLN/USD'
             ,'^FVX':'5_YB','RUB=X':'USD/RUB','PL=F':'Platinum',
             'SI=F':'Silver','NG=F':'Natural Gas','ZR=F':'Rice Futures',
             'ZS=F':'Soy Futures','KE=F':'KC HRW Wheat Futures'}

# Definicje do Korelacji
comm_dict2 = {k: v for k, v in comm_dict.items() if k != '^DJI'}

#Źródło danych do korelacji
@st.cache_data
def cor_tab(past):        
    df_list = []
    col_n = {'Close': 'DJI30'}
    t = yf.Ticker('^DJI')
    x = pd.DataFrame(t.history(period='1d', start='2003-12-01', end = today))
    x1 = x.reset_index()
    x2 = pd.DataFrame(x1['Close'][-past:])
    x2.rename(columns = col_n, inplace=True)
    x2 = pd.DataFrame(x2.reset_index(drop=True)) 
    for label, name in comm_dict2.items(): 
        col_name = {'Close': name}
        t1 = yf.Ticker(label)
        y1 = pd.DataFrame(t1.history(period='1d', start='2003-12-01', end = today)) 
        y1.reset_index()
        y2 = y1[['Close']][-past:]
        y2 = pd.DataFrame(y2.reset_index(drop=True))
        y2.rename(columns = col_name, inplace=True)
        m_tab = pd.concat([x2, y2], axis=1)
        df_list.append(m_tab)
        cor_df = pd.concat(df_list, axis=1)
        cor_df = cor_df.T.drop_duplicates().T
        cor_df.fillna(0)
        cor_data = cor_df.to_pickle('cor_data.pkl')
   
cor_tab(1000)  

# Pobieranie danych
def comm_f(comm):
    global df_c1
    for label, name in comm_dict.items():
        if name == comm:
            df_c = pd.DataFrame(yf.download(f'{label}', start='2000-09-01', end = today,interval='1d'))
            df_c1 = df_c.reset_index()
           
    return df_c1   

# Dane historyczne                    
def comm_data(comm):
    global Tab_his1
    shape_test=[]
    sh = df_c1.shape[0]
    start_date = df_c1.Date.min()
    end_date = df_c1.Date.max()
    close_max = "{:.2f}".format(df_c1['Close'].max())
    close_min = "{:.2f}".format(df_c1['Close'].min())
    last_close = "{:.2f}".format(df_c1['Close'].iloc[-1])
    v = (comm, sh, start_date,end_date,close_max,close_min,last_close)
    shape_test.append(v)
    Tab_length = pd.DataFrame(shape_test, columns= ['Name','Rows', 'Start_Date', 'End_Date','Close_max','Close_min','Last_close'])   
    Tab_his = Tab_length[['Start_Date','End_Date','Close_max','Close_min','Last_close']]
    Tab_his['Start_Date'] = Tab_his['Start_Date'].dt.strftime('%Y-%m-%d')
    Tab_his['End_Date'] = Tab_his['End_Date'].dt.strftime('%Y-%m-%d')
    #Tab_his1 = Tab_his.T
    #Tab_his1.rename(columns={0: "Details"}, inplace=True)
    
    return st.dataframe(Tab_his, hide_index=True)

st.title('4 Trends estimation models')
with st.form("<gen_form>"):
    comm = st.radio('What do you want to analyse today ?', list(comm_dict.values()), horizontal=True,  key = "<gen1>")
    submitted = st.form_submit_button("Submit")

comm_f(comm)

st.subheader(f'General metrics of {comm}')
comm_data(comm)
    
# Arima - model - prognoza trendu
@st.cache_data
def Arima_f(comm, size_a):
    data = np.asarray(df_c1['Close'][-300:]).reshape(-1, 1)
    p = 10
    d = 0
    q = 5
    n = size_a

    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit(method_kwargs={'maxiter': 3000})
    model_fit = model.fit(method_kwargs={'xtol': 1e-6})
    fore_arima = model_fit.forecast(steps=n)  
    
    arima_dates = [datetime.today() + timedelta(days=i) for i in range(0, size_a)]
    arima_pred_df = pd.DataFrame({'Date': arima_dates, 'Predicted Close': fore_arima})
    arima_pred_df['Date'] = arima_pred_df['Date'].dt.strftime('%Y-%m-%d')
    arima_df = pd.DataFrame(df_c1[['Date','High','Close']][-500:])
    arima_df['Date'] = arima_df['Date'].dt.strftime('%Y-%m-%d')
    arima_chart_df = pd.concat([arima_df, arima_pred_df], ignore_index=True)
    x_ar = (list(arima_chart_df.index)[-1] + 1)
    arima_chart_dff = arima_chart_df.iloc[x_ar - 30:x_ar]
    
    fig_ar = px.line(arima_chart_dff, x='Date', y=['High', 'Close', 'Predicted Close'], color_discrete_map={
                  'High': 'yellow', 'Close': 'black', 'Predicted Close': 'red'}, width=900, height=500)
    fig_ar.add_vline(x = today,line_width=1, line_dash="dash", line_color="green")
    fig_ar.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_ar, use_container_width=True)      
    
# Estymacja przy pomocy Linear Regression
@st.cache_data
def LR_f(comm, num1):
    size = num1
        
    X = df_c1['Open'].values.reshape(-1, 1)
    y = df_c1['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    future_dates = [datetime.today() + timedelta(days=i) for i in range(0, size)]
    future_open = [[open_val] for open_val in df_c1['Open'].tail(size)]
    predicted_close = model.predict(future_open)
    predicted_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predicted_close})

    predicted_df['Date'] = predicted_df['Date'].dt.strftime('%Y-%m-%d')
    LR_tab = pd.DataFrame(predicted_df)
    df_c12 = pd.concat([df_c1, LR_tab], ignore_index=True)

    xbb = (list(df_c12.index)[-1] + 1)
    df_c13 = df_c12.iloc[xbb - 30:xbb]
    
    fig_lr = px.line(df_c13,x='Date', y=['High','Close','Predicted Close'],color_discrete_map={
                 'High':'#0d0887','Close':'#f0f921','Predicted Close':'#d62728'}, 
              width=900, height=500) 
    fig_lr.add_vline(x = today,line_width=1, line_dash="dash", line_color="green")
    fig_lr.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_lr, use_container_width=True)
           
# SVR regression model
@st.cache_data
def DES_f(comm, num2):
    size = num2
    data = df_c1['Close'].values[-1001:]
    train_data = data[:-size]
    model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(size)
           
    des_dates = [datetime.today() + timedelta(days=i) for i in range(0, size)]
    pred_close = model_fit.forecast(size)
    pred_des = pd.DataFrame({'Date': des_dates, 'Predicted Close': pred_close})
    DES_c12 = pd.concat([df_c1, pred_des], ignore_index=True)
        
    xbcc = (list(DES_c12.index)[-1] + 1)
    DES_c13 = DES_c12.iloc[xbcc - 30:xbcc]
    DES_c13['Date'] = pd.to_datetime(DES_c13['Date'], format='%Y-%m-%d')
    
    fig_des = px.line(DES_c13, x='Date', y=['High', 'Close', 'Predicted Close'], color_discrete_map={
                  'High': 'purple', 'Close': 'yellow', 'Predicted Close': '#d62728'}, width=900, height=500)
    fig_des.add_vline(x = today,line_width=1, line_dash="dash", line_color="black")
    fig_des.update_layout(xaxis=None, yaxis=None)    
    st.plotly_chart(fig_des, use_container_width=True)  

# Gausian model    
@st.cache_data
def gaus_f(comm,horizon):
    global var1
    g_data = pd.read_pickle('cor_data.pkl')
    correlation = g_data.corr(numeric_only=False)[comm].sort_values(ascending=False)
    select_cor = pd.DataFrame(correlation)
    var1 = select_cor[1:2].index.values[0]   
     
    for label, name in comm_dict.items():
        if name == var1:
            xx = pd.DataFrame(yf.download(f'{label}', start='2000-09-01', end = today,interval='1d'))
            xx1 = xx.reset_index()
        
    for label, name in comm_dict.items():
        if name == comm:
            yy = pd.DataFrame(yf.download(f'{label}', start='2000-09-01', end = today,interval='1d'))
            yy1 = yy.reset_index()
    
    scope = 300
    x = np.array(xx1['Close'][-scope:]).reshape(-1, 1)  
    y = np.array(yy1['Close'][-scope:])  

    kernel = RBF(length_scale=1.0)  
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(x, y)
    
    x_pred = np.array(xx1['Close'][-scope+1:-scope+horizon+1]).reshape(-1, 1) 
    y_pred, std = model.predict(x_pred, return_std=True)
    
    ga_dates = [datetime.today() + timedelta(days=i) for i in range(0, horizon)]
    ga_pred_df = pd.DataFrame({'Date': ga_dates, 'Predicted Close': y_pred})
    ga_pred_df['Date'] = ga_pred_df['Date'].dt.strftime('%Y-%m-%d')
    ga_df = pd.DataFrame(yy1[['Date','High','Close']][-500:])
    ga_df['Date'] = ga_df['Date'].dt.strftime('%Y-%m-%d')
    ga_chart_df = pd.concat([ga_df, ga_pred_df], ignore_index=True)
    x_ga = (list(ga_chart_df.index)[-1] + 1)
    ga_chart_dff = ga_chart_df.iloc[x_ga - 30:x_ga]
    
    fig_ga = px.line(ga_chart_dff, x='Date', y=['High', 'Close', 'Predicted Close'], color_discrete_map={
                  'High': 'yellow', 'Close': 'limegreen', 'Predicted Close': 'red'}, width=900, height=500)
    fig_ga.add_vline(x = today,line_width=1, line_dash="dash", line_color="black")
    fig_ga.update_layout(xaxis=None, yaxis=None)
    fig_ga.add_annotation(x=today , y= ['Predicted Close'], text= f'Prediction computed based on the best correlated asset - {var1}', showarrow=False)
    st.plotly_chart(fig_ga, use_container_width=True)
        
st.subheader('Trend estimations based on regressions')
st.divider()
col1, col2 = st.columns(2)
with col1:   
    st.write(f'{comm} - Linear regression trend estimation')
    num1 = st.number_input('Enter the number of days for trend estimation', value=10, key = "<esti11>")
    LR_f(comm, num1)    

with col2:       
    st.write(f'{comm} - DES trend estimation')
    num2 = st.number_input('Enter the number of days for trend estimation', value=10, key = "<esti12>")
    DES_f(comm, num2)

st.subheader('Trend estimations based on econometric models')
st.divider()
col3, col4 = st.columns(2)    
with col3:
    st.write(f'{comm} - Arima model trend prediction')
    size_a = st.radio('Prediction for ... days ?: ', [5,4,3,2,1], horizontal=True, key = "<arima21>")
    Arima_f(comm, size_a)    
    with st.expander("Arima parameters"):
        st.write("Autoregression (AR): This parameter tells how much the current value depends on the previous values. If you increase this parameter, the model will focus more in recent trends in the data. If you decrease it, the model will be less dependent on recent trends and more focused on the overall direction of the data. Moving Average (MA): This parameter tells how much the current value depends on forecasting errors from the past. If you increase this parameter, the model will focus more on its recent errors and try to correct them. If you decrease it, the model will be less focused on its errors and more focused on the actual data. Integration (I): This parameter tells how many times we need to differentiate the data to make it stationary (i.e., its statistical properties do not change over time). If you increase this parameter, the model will focus more on long-term trends in the data. If you decrease it, the model will be more focused on short-term fluctuations.")
                      
with col4:
    st.write(f'{comm} - Gaussian model trend prediction')
    horizon = st.radio('Prediction for ... days ?:', [10,9,8,7,6,5,4,3,2,1], horizontal=True, key = "<ga1>")
    cor_tab(1000)    
    gaus_f(comm, horizon)
    with st.expander('Gaussian model parameters'):
        st.write('C: The C parameter is a regularization parameter that helps fit the model to the data. A high C value allows the model to fit to each data point in the training set, which can lead to overfitting. A low C value allows the model to tolerate errors and can lead to underfitting. Therefore, choosing the right C value is key to the effectiveness of the model. Epsilon: The epsilon parameter defines the error margin that is accepted by the model. All predictions that are within this margin from the actual values are treated as correct by the model. Increasing epsilon results in greater tolerance for errors but can lead to inaccurate predictions. Decreasing epsilon results in less tolerance for errors but can lead to overfitting. Kernel: The kernel parameter determines the function used to transform the input data. Different kernels can be used depending on the nature of the data and the problem you are trying to solve. The most popular kernels are ‘linear’, ‘poly’, ‘rbf’, and ‘sigmoid’.')

st.subheader('Results of my own (D+1) LSTM Prediction Models')
col5, col6 = st.columns(2)    
with col5:
    st.write('USD/PLN exchange rate (D+1) predictions - last 50 days')
    st.divider()
    val_USD = pd.read_excel('LSTM_mv.xlsx', sheet_name='D1_USD')
    val_USDD1 = val_USD[['Date','USD/PLN','Day + 1 Prediction']].iloc[:-1]

    fig_USDD1 = px.line(val_USDD1[-50:], x='Date', y=['USD/PLN','Day + 1 Prediction'], color_discrete_map={
                      'USD/PLN': 'mediumseagreen', 'Day + 1 Prediction': 'dodgerblue'}, width=1000, height=500)
    fig_USDD1.update_layout(xaxis=None, yaxis=None)    
    st.plotly_chart(fig_USDD1, use_container_width=True)  
with col6:
    st.write('EUR/PLN exchange rate (D+1) predictions - last 50 days')
    st.divider()
    val_EUR = pd.read_excel('LSTM_mv.xlsx', sheet_name='D1_EUR')
    val_EURD1 = val_EUR[['Date','EUR/PLN','Day + 1 Prediction']].iloc[:-1]

    fig_EURD1 = px.line(val_EURD1[-50:], x='Date', y=['EUR/PLN','Day + 1 Prediction'], color_discrete_map={
                      'EUR/PLN': 'tomato', 'Day + 1 Prediction': 'dodgerblue'}, width=1000, height=500)
    fig_EURD1.update_layout(xaxis=None, yaxis=None)    
    st.plotly_chart(fig_EURD1, use_container_width=True)  