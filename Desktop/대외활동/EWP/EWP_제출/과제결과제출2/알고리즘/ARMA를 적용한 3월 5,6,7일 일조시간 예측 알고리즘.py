
# coding: utf-8

# # 시계열 모델을 이용한 일조 예측

# ### 분석 순서 
# 1. 모듈 import
# 2. 데이터 read 및 전처리
# 3. ARMA

# ### 1. 모듈 import

# In[136]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas import DataFrame, datetime
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ### 2. 데이터 read 및 전처리

# In[137]:


wt_train = pd.read_excel("F:\\2차 제출 데이터\\A_site_fin.xlsx", sheet_name=0)
wt_test = pd.read_excel("F:\\2차 제출 데이터\\0305_0307_update_15min.xlsx")          

wt_train = wt_train.set_index('time')
wt_test = wt_test.set_index('Time')


# #### 2-1. 데이터 타입 변환  

# In[138]:


wt_test = wt_test.astype('float64')
wt_train = wt_train.astype('float64')


# #### 2-2. 데이터 셋 정의  

# In[139]:


endog = wt_train[['sunshine','temp','dailyAccRain','wind_velocity','airpressure','sealevelpressure','humidity',
                 'cloud']]
exog = wt_test[['temp','dailyAccRain','wind_velocity','airpressure','sealevelpressure','humidity','cloud']]

data = endog.append(exog, sort=False)


# In[140]:


endog = data.loc[:'2019-02-23 17:45:00',:]
exog = data.loc['2019-03-05 09:00:00':, :]


# #### 2-3. X, Y변수 설정  

# In[141]:


y = endog['sunshine']
x = endog.drop(endog.columns[0], axis=1)    #sunshine, SolarRad 제외


# In[142]:


exog = exog.drop(exog.columns[0], axis=1)


# ### 3. ARMA 모형

# In[143]:


model = sm.tsa.ARMA(y, (1,1), exog=x)
r = model.fit()
print(r.summary())


# #### 3-1. ACF plot

# In[144]:


plt.figure(figsize=(30,15))
sm.tsa.graphics.plot_acf(r.resid);


# #### 3-2. PACF plot  

# In[99]:


plt.figure(figsize=(30,15))
sm.tsa.graphics.plot_pacf(r.resid);


# #### 3-3. 일조 예측

# In[145]:


yhat = r.predict(start=3024, end=3131, exog=exog)


# In[146]:


yhat


# In[147]:


#exog의 index를 칼럼으로 빼낸 후, index 칼럼을 time 칼럼으로 사용
exog1 = exog.reset_index()
yhat_ = pd.DataFrame(yhat)
yhat_ = yhat_.reset_index()
result = pd.DataFrame({"pred_sunshine":yhat_[0],"time":exog1['index']})


# In[148]:


result


# #### 3-4. 결과값 추세 확인  

# In[149]:


plt.figure(figsize=(8,3))
plt.plot(result['pred_sunshine'])


# In[49]:


result.to_excel('F:\\2차 제출 데이터\\sunshine_forecast.xlsx', 'sheet1',index=False, engine='xlsxwriter')

