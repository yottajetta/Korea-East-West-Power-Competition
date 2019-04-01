
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, Activation,Dropout, Conv1D, MaxPooling1D, BatchNormalization, Permute, TimeDistributed,Reshape
from tensorflow.python.keras.optimizers import adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


# # 데이터 들여오기 

# In[49]:


AN=pd.read_excel('F:\\2차 제출 데이터\\A_site_fin.xlsx')
weather=pd.read_excel('F:\\2차 제출 데이터\\0306_0307_15min.xlsx')
AN=AN.set_index('time')
weather=weather.set_index('Time')


# In[3]:


A=AN[['temp','dailyAccRain', 'wind_velocity', 'humidity','airpressure', 
      'sealevelpressure','sunshine', 'SolarRad','cloud']]
w=weather[['temp','dailyAccRain', 'wind_velocity', 'humidity','airpressure', 
      'sealevelpressure','cloud']]


# # 데이터 정규화 

# In[59]:


#2018년 3월 기상청 종관기상관측 데이터 정규화
scaler1=MinMaxScaler(feature_range=(0,1))
scaler1=scaler1.fit(A[['SolarRad']])
scaled1=scaler1.fit_transform(A[['SolarRad']])
scaler2=MinMaxScaler(feature_range=(0,1))
scaled2=scaler2.fit_transform(A[['temp','dailyAccRain', 'wind_velocity', 'humidity',
       'airpressure', 'sealevelpressure','cloud']])
#예보데이터 정규화(일사량의 예측에 쓰일 기상 데이터)
scaled3=scaler2.fit_transform(weather[['temp','dailyAccRain', 'wind_velocity', 'humidity','airpressure', 
      'sealevelpressure','cloud']])
#데이터 활용을 위해 dataframe으로 형변환
scaled1=pd.DataFrame(scaled1)
scaled2=pd.DataFrame(scaled2)
scaled3=pd.DataFrame(scaled3)
#훈련 데이터 연결 및 변수명 부여
scaled=pd.concat([scaled1, scaled2], axis=1)
scaled.columns=['SolarRad','Temp','dailyAccRain', 'wind_velocity',
                'humidity','airpressure', 'sealevelpressure','cloud']
#test 데이터 변수명 부여
scaled3.cloumns=['temp','dailyAccRain', 'wind_velocity', 'humidity','airpressure', 
      'sealevelpressure','cloud']


# # 종속변수, 독립변수 설정 및 데이터 형태 변환 

# In[60]:


#X, y값 설정
yB=scaled[['SolarRad']]
xB=scaled[['Temp','dailyAccRain', 'wind_velocity', 'humidity',
       'airpressure', 'sealevelpressure','cloud']]

#CNN-LSTM 모델에 Input으로 넣기 위한 Reshape
#기상청 종관기상관측 데이터 Reshape
xB=xB.values.reshape(int(len(scaled)/36),36,7,1)
yB=yB.values.reshape(int(len(scaled)/36),36)
#Weather Undergroun 데이터 Reshape
wA=scaled3.values.reshape(int(len(scaled3)/36),36,7,1)

#이 이하 내용은 종관기상관측데이터를 활용하여 모델 검증을 거칠 때 사용한 코드입니다.
#현재는 weather underground를 활용한 3월 5, 6, 7일 일사량 예측이므로 주석처리 합니다.
#xB_train, xB_test, yB_train, yB_test=train_test_split(xB,yB, test_size=0.2, shuffle=False)
#print(xB_train.shape,xB_test.shape,yB_train.shape,yB_test.shape)


# # Keras를 활용한 CNN-LSTM 모델 생성 및 학습 

# In[61]:


model1=Sequential() #input_shape=(36=오전 9시부터 5시까지의 15분 단위 데이터 갯수 ,7 = 일사량 예측을 위한 변수 갯수)
model1.add(TimeDistributed(Conv1D(32, kernel_size=2, activation='relu'), input_shape=(xB.shape[1],xB.shape[2],1)))
model1.add(BatchNormalization()) #배치 정규화를 통한 과적합 방지
model1.add(Dropout(0.3)) #과적합을 막기위한 dropout
model1.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
#컨볼루션 연산을 통해 생성된 데이터의 지역적 특징 중 가장 큰 값을 반영하여 특징을 함축하는 MaxPooling층을 사용하여 지역적 특징을 연산합니다.
#strides가 2이므로 입력 변수 갯수가 반으로 줄어듭니다.
model1.add(Reshape((36,96), input_shape=(36,3,32))) #LSTM에 넣기 위해 3차원 데이터를 2차원으로 변환.
model1.add(LSTM(32, input_shape=(36,96),return_sequences=True)) #stacked LSTM(2층 짜리) 구현
model1.add(Activation('relu')) #활성화 함수로 ReLu적용
model1.add(Dropout(0.3))#과적합을 막기위한 dropout
model1.add(LSTM(32)) #두 번째 LSTM
model1.add(Activation('relu'))#활성화 함수로 ReLu적용
model1.add(Dropout(0.3))#과적합을 막기위한 dropout
model1.add(Dense(36))# 하루 단위 결과 출력 
model1.compile(loss='mse', optimizer='adam') #손실함수 MSE, 최적화 함수 ADAM 적용
model1.summary()
model1.fit(xB, yB, epochs=370, verbose=1) 
#기존의 기상청 데이터와 일사량데이터를 훈련데이터로 하여 학습 반복 횟수 370으로 설정하여 모델 학습 진행.


# In[62]:


pred1=model1.predict(scaled3) #weather underground 데이터로 일사량 예측
y_hat1=pd.DataFrame(pred1) #원활한 데이터 확인을 위하여 dataframe으로 변환

#역표준화를 위한 MinMaxScaler 생성
scaler1=MinMaxScaler(feature_range=(0,1))
scaler1=scaler1.fit(A[['SolarRad']])
scaled1=scaler1.fit_transform(A[['SolarRad']])
#생성된 예측값 역표준화
yhat1=scaler1.inverse_transform(y_hat1)
#아래 코드는 기상청 종관기상관측을 이용하여 모델 검증시 train의 예측 값 검증을 위하여 test의 일사량을 역표준화 하는 코드입니다.
y=scaler1.inverse_transform(yB_test)


# In[63]:


#기상청 종관 기상 관측 데이터를 활용하여 train, test 검증시 활용한 RMSE코드.
rmse=np.sqrt(mse(y_hat1, y))
print('Test RMSE: %.3f' % rmse)


# In[70]:


#기상청 종관 기상 관측 데이터를 활용한 모델 검증시 예측된 예측값과 실제 값을 그래프로 확인
plt.figure(figsize=(7,2))
plt.plot(yhat1.ravel(), 'g')
plt.plot(y.ravel(), 'r')

