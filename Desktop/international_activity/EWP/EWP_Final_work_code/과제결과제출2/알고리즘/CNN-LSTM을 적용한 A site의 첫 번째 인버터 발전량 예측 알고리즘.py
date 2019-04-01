
# coding: utf-8

# # A site 첫 번째 인버터 발전량 예측
# 아래 알고리즘을 모든 사이트의 모든 인버터에 동일하게 적용하였습니다. 데이터 파일과 변수명만 바꾸어 예측이 가능합니다. 
# 따라서 대표적인 하나의 알고리즘 파일을 첨부합니다.

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM, Activation,Dropout, Conv1D, MaxPooling1D,Conv2D, MaxPooling2D,Flatten, BatchNormalization, Permute, TimeDistributed,Reshape, RepeatVector
from tensorflow.python.keras.optimizers import adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


# # 데이터 들여오기 

# In[31]:


AN=pd.read_excel('F:\\2차 제출 데이터\\A_site_fin.xlsx')
test=pd.read_excel('F:\\2차 제출 데이터\\0306_0307_15min.xlsx')
AN=AN.set_index('time')


# In[32]:


A=AN[['inv1ACPower','temp','dailyAccRain', 'wind_velocity', 'humidity',
       'airpressure', 'sealevelpressure','sunshine', 'SolarRad','cloud']]


# # A site의 인버터 발전량, 기상 데이터, Weather underground의 3월 5,6,7일 예보데이터 표준화 

# In[35]:


#A site의 첫 번째 인버터 발전량 값을 0과 1사이 값으로 표준화
scaler1=MinMaxScaler(feature_range=(0,1))
scaler1=scaler1.fit(A[['inv1ACPower']])
scaled1=scaler1.fit_transform(A[['inv1ACPower']])
#기상청 데이터 및 Weather Underground 데이터 표준화
scaler2=MinMaxScaler(feature_range=(0,1))
scaled2=scaler2.fit_transform(A[['temp',
'dailyAccRain', 'wind_velocity', 'humidity',
       'airpressure', 'sealevelpressure','sunshine', 'SolarRad','cloud']])
scaled3=scaler2.fit_transform(test[['temp','dailyAccRain', 'wind_velocity', 'humidity','airpressure', 
      'sealevelpressure','cloud','sunshine', 'SolarRad']])
#원활한 활용을 위하여 표준화된 값을 array에서 dataframe으로 변환
scaled1=pd.DataFrame(scaled1)
scaled2=pd.DataFrame(scaled2)
scaled3=pd.DataFrame(scaled3)
#표준화된 인버터 발전량 데이터와 기상청 기상 데이터를 결합시킨 scaled 확인
scaled=pd.concat([scaled1, scaled2], axis=1)


# In[36]:


#생된 데이터 프레임에 변수명 부여
scaled.columns=['inv1ACPower','temp','dailyAccRain', 'wind_velocity', 'humidity',
       'airpressure', 'sealevelpressure','sunshine', 'SolarRad','cloud']


# # 독립 변수, 종속변수 선정 및 CNN-LSTM 입력을 위한 데이터 형태변환. 

# In[37]:


#독립 변수, 종속변수 설정
yA=scaled[['inv1ACPower']]
xA=scaled[['temp',
'dailyAccRain', 'wind_velocity', 'humidity',
       'airpressure', 'sealevelpressure','sunshine', 'SolarRad','cloud']]
#독립변수와 종속변수를 나누어 데이터 형변환
#(발전량 데이터의 총 일 수(Number of the day), 36(오전 9시부터 5시 45분 까지의 15분 단위 데이터 갯수), 9(변수의 갯수), 축)
xA=xA.values.reshape(int(len(scaled)/36),36,9,1)
yA=yA.values.reshape(int(len(scaled)/36),36)
test=scaled3.values.reshape(int(len(scaled3)/36),36,9,1)

#이 이하 내용은 기존의 인버터 발전량과 종관기상관측데이터를 활용하여 모델 검증을 거칠 때 사용한 코드입니다.
#weather underground의 예보 데이터를 활용하여 실제 인버터 발전량을 예측하는 모델에는 사용하지 않습니다.
#xA_train, xA_test, yA_train, yA_test=train_test_split(xA,yA, test_size=0.2, shuffle=False)
#print(xA_train.shape,xA_test.shape,yA_train.shape,yA_test.shape)


# # Keras를 활용한 CNN-LSTM 모델 생성 및 학습 

# In[38]:


#순차적으로 모델을 적용하므로 sequential() 선언
model=Sequential() 
model.add(TimeDistributed(Conv1D(32, kernel_size=2, activation='relu'), input_shape=(xA.shape[1],xA.shape[2],1)))
#활성 함수= ReLu를 사용합니다.
#input_shape=(발전량 데이터의 총 일 수(Number of the day), 36(오전 9시부터 5시 45분 까지의 15분 단위 데이터 갯수), 9(변수의 갯수), 축)
model.add(BatchNormalization()) #모델 과적합을 방지하기 위한 배치 정규화.
model.add(Dropout(0.3))# 모델 과적합을 방지하기 위한 Drop out
model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2))) 
#컨볼루션 연산을 통해 생성된 데이터의 지역적 특징 중 가장 큰 값을 반영하여 특징을 함축하는 MaxPooling층을 사용하여 지역적 특징을 연산합니다.
#strides가 2이므로 입력 변수 갯수가 반으로 줄어듭니다.
model.add(Reshape((36,128), input_shape=(36,4,32))) #LSTM에 넣기 위해 3차원 데이터를 2차원으로 변환.
model.add(LSTM(32, input_shape=(36,96),return_sequences=True))#stacked LSTM(2층 짜리) 구현
model.add(Activation('relu'))#활성화 함수로 ReLu적용
model.add(Dropout(0.3))#과적합을 막기위한 dropout
model.add(LSTM(32)) #두 번째 LSTM
model.add(Activation('relu'))#활성화 함수로 ReLu적용
model.add(Dropout(0.3))#과적합을 막기위한 dropout
model.add(Dense(36))#하루 단위 결과 출력 
model.compile(loss='mse', optimizer='adam')#손실함수 MSE, 최적화 함수 ADAM 적용
model.summary()
model.fit(xA, yA, epochs=370, verbose=1) 
#기존의 A site 인버터 데이터와 기상청 종관기상관측데이터를 훈련 데이터로 합니다.
#학습 횟수 370회로 모델 학습을 진행합니다.


# # Weather Underground의 예보데이터 + 예측된 일사량, 일조시간 테스트 데이터로
# # 15분 단위 인버터 발전량 데이터 예측

# In[22]:


#기상 예보데이터로 인버터 발전량 예측
pred=model.predict(test)  
y_hat=pd.DataFrame(pred)# 예측된 인버터 발전량을 array에서 dataframe으로 변환.


# ## 생성된 예측값 역표준화 단계 

# In[23]:


#기존에 인버터 발전량을 표준화했던 scaler1을 재호출하여 스케일러를 생성합니다.
scaler1=MinMaxScaler(feature_range=(0,1))
scaler1=scaler1.fit(A[['inv1ACPower']])
scaled1=scaler1.fit_transform(A[['inv1ACPower']])


# In[24]:


#생성된 scaler1을 활용하여 역표준화를 진행합니다.
yhat=scaler1.inverse_transform(y_hat)
#이하 코드는 모델 검증시 test 데이터 역표준화를 위해 사용한 코드입니다.
#y=scaler1.inverse_transform(yA_test)


# ### 모델 검증시 사용한 RMSE와 그래프 Plot
# 이하 코드는 모델 검증을 위하여 테스트 데이터를 활용한 예측값과 실제 테스트 데이터 사이의 RMSE를 구하는 것입니다.

# In[26]:


rmse=np.sqrt(mse(pred, yA_test))
mae=mae(pred, yA_test)
print('Test RMSE: %.3f' % rmse, 'Test MAE: %.3f' % mae)


# In[30]:


plt.figure(figsize=(10,3))
plt.plot(yhat.ravel(), 'g')
plt.plot(y.ravel(), 'r')


# train=Input(shape=(xA_train.shape[1],xA_train.shape[2],1))
# 
# conv1=Conv2D(32, kernel_size=(3,9), use_bias=False)(train)
# inner=BatchNormalization()(conv1)
# inner=Activation('relu')(inner)
# inner=Dropout(0.3)(inner)
# inner=MaxPooling2D(pool_size=(2,1), strides=2)(inner)
# inner=Dense(32)(inner)
# inner=Reshape((17,32), input_shape=(17,1,32))(inner)
# inner=GRU(32, input_shape=(17,32),return_sequences=True)(inner)
# inner=Activation('relu')(inner)
# inner=Dropout(0.3)(inner)
# inner=GRU(32)(inner)
# inner=Activation('relu')(inner)
# inner=Dropout(0.3)(inner)
# output=Dense(36, activation='relu')(inner)
# model=Model(inputs=train, outputs=output)
# #model.compile(loss='mse', optimizer='adam')
# model.summary()
# plot_model(model, to_file='model_graph.png')
