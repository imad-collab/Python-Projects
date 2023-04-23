#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.layers import LSTM,Dropout,Dense
from keras.layers import Flatten
from sklearn.metrics import mean_squared_error


# In[2]:


df=pd.read_csv('Reliance_Stock.csv')
df


# In[3]:


sns.boxplot(x=df['Open'])
plt.show()


# In[4]:


sns.boxplot(x=df['High'])
plt.show()


# In[5]:


sns.boxplot(x=df['Close'])
plt.show()


# In[6]:


sns.boxplot(x=df['Low'])
plt.show()


# In[7]:


sns.boxplot(x=df['Volume'])
plt.show()


# In[8]:


from datetime import datetime

date_string = '01-01-2015'
date_object = datetime.strptime(date_string, '%d-%m-%Y')

print(date_object)


# In[9]:


df1 = pd.DataFrame(df)
dict = {"Open" : "open", "High" : "high", "Low" : "low", "Close" : "close", "Adj Close" : "adj_close", "Volume" : "volume"}
df1.rename(columns = dict, inplace = True)


# In[10]:


df.sort_values(by='open',ascending=True)


# In[11]:


df.sort_values(by='close',ascending=True)


# In[12]:


df.sort_values(by='high',ascending=True)


# In[13]:


df.sort_values(by='low',ascending=True)


# In[14]:


df.corr(method='pearson')['open']


# In[15]:


df1=df.select_dtypes(include=['float64','int64'])


# In[16]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[17]:


sns.boxplot(x='open',y='close',data=df)
plt.show()


# In[55]:


sns.boxplot(x='high',y='low',data=df)
plt.show()


# In[56]:


sns.boxplot(x='adj_close',y='volume',data=df)
plt.show()


# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.scatter(x = df['open'], y = df['close'], color = 'green')
plt.xlabel('Period', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.show()


# In[58]:


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.scatter(x = df['adj_close'], y = df['close'], color = 'green')
plt.xlabel('Period', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.show()


# In[59]:


fig,ax= plt.subplots(figsize=(10,6))
ax.scatter(df['high'], df['low'],color='red')
ax.set_xlabel('high')
ax.set_ylabel('low')
plt.show()


# In[60]:


fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['adj_close'], df['volume'])
ax.set_xlabel('adj_close')
ax.set_ylabel('volume')
plt.show()


# In[61]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df.hist(figsize=(8, 8), bins=50, xlabelsize=4, ylabelsize=4); 


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(df[['open']], color='b', bins=100, hist_kws={'alpha': 0.4});
plt.show()


# In[63]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(df[['high']], color='g', bins=100, hist_kws={'alpha': 0.4});
plt.show()


# In[98]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df)
plt.show()


# In[51]:



df.dtypes


# In[20]:


df


# In[101]:


plt.figure(figsize=(10,5))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c


# In[105]:


print(sns.displot(x='Open',y='Close',data=df))
print(sns.displot(x='High',y='Low',data=df))
print(sns.displot(x='Adj Close',y='Adj Close',data=df))


# In[107]:


print(sns.countplot(x='Low',data=df))
plt.show()


# In[108]:


print(sns.countplot(x='High',data=df))
plt.show()


# In[109]:


print(sns.countplot(x='Close',data=df))
plt.show()


# In[110]:


print(sns.countplot(x='Open',data=df))
plt.show()


# In[111]:


sns.jointplot(x='Open',y='Close',data=df)
plt.show()


# In[ ]:


sns.jointplot(x='High',y='Low',data=df)
plt.show()


# In[112]:


plt.figure(figsize = (14,5), dpi = 80)
print(df["Open"].plot(ylim = [0,2900], c = 'green'))


# In[113]:


plt.figure(figsize = (14,5), dpi = 80)
print(df["Close"].plot(ylim = [0,2900], c = 'red'))


# In[ ]:


plt.figure(figsize = (14,5), dpi = 80)
print(df["High"].plot(ylim = [0,2900], c = 'blue'))


# In[114]:


plt.figure(figsize = (14,5), dpi = 80)
print(df["Low"].plot(ylim = [0,2900], c = 'pink'))


# In[117]:


plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.kdeplot(df["Open"], df["Volume"], cmap = "cool_d")
plt.title("KdePlot of Open")

plt.subplot(2,2,2)
sns.kdeplot(df["Close"], df["Volume"], cmap = "cool_d")
plt.title("KdePlot of Close")
           
plt.subplot(2,2,3)
sns.kdeplot(df["High"], df["Volume"], cmap = "cool_d")
plt.title("KdePlot of High")
           
plt.subplot(2,2,4)
sns.kdeplot(df["Low"], df["Volume"], cmap = "cool_d")
plt.title("KdePlot of Low")


# In[119]:


plt.figure(figsize=(12,10))
plt.subplot(2, 2, 1)
df.Close.plot(kind='line')

plt.title('Close')
plt.ylabel('Value')
plt.xlabel('Year')


plt.subplot(2, 2, 2)
df.Volume.plot(kind='line')

plt.title('Volume')
plt.ylabel('Shares')
plt.xlabel('Year')

plt.subplot(2, 2, 3)
df.High.plot(kind='line')

plt.title('High')
plt.ylabel('Value')
plt.xlabel('Year')

plt.subplot(2, 2, 4)
df.High.plot(kind='line')

plt.title('Low')
plt.ylabel('Value')
plt.xlabel('Year')

plt.tight_layout()
plt.show()


# # Model Building

# # RNN Model

# In[21]:


df=pd.read_csv('Reliance_Stock.csv')


# In[22]:


dataset_train=pd.read_csv('Reliance_Stock.csv')
dataset_train


# In[23]:


#Set Target Variable
output_var = pd.DataFrame(df['Adj Close'])
#Selecting the Features
features = ['Open', 'High', 'Low', 'Volume']


# In[24]:


#Scaling
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)
feature_transform.head()


# In[25]:


#Splitting to Training set and Test set
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()


# In[26]:


#Process the data for LSTM
trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])


# In[27]:


#Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
plot_model(lstm, show_shapes=True, show_layer_names=True)


# In[ ]:


history=lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)


# In[53]:


#LSTM Prediction
y_pred= lstm.predict(X_test)


# In[ ]:





# In[54]:


#LSTM Prediction
y_pred= lstm.predict(X_test)


# In[55]:


#Predicted vs True Adj Close Value – LSTM
plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.title('Prediction by LSTM')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()


# In[56]:


train_set=df.iloc[:,4:5].values
train_set


# In[57]:


#Print the shape of Dataframe  and Check for Null Values
print('Dataframe Shape: ', df. shape)


# In[58]:


#The data is in 2D array format
# The RNN expects the data to be scaled.
#That means we have to fit the data in between some values like 0 to 1.
# This can be done by using MinMax Scale


# In[59]:


#units=50 means 50 nodes
#return_sequences=True means it is going to return same number of outputs to the next layer.
#input_shape=(None,1) means only 1 column of data is being given as input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential([LSTM(units=50,return_sequences=True,input_shape=(None,1)),
                  LSTM(units=50,return_sequences=True),Dense(1,activation='sigmoid')])


# In[60]:


model.compile(optimizer='adam',loss='mean_squared_error')


# In[61]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
inputs=sc.fit(train_set)
inputs=sc.transform(train_set)


# In[62]:


inputs


# In[63]:


pred_values=model.predict(inputs)
pred_values


# In[64]:


pred_values.shape


# In[65]:


today_price=float(input('enter the today stockprice: '))


# In[66]:


arr=np.array(today_price).reshape(-1,1)
arr1=sc.transform(arr)


# In[67]:


arr1


# In[68]:


arr1=arr1.reshape(-1,1,1)
result=model.predict(arr1)


# In[69]:


result=np.reshape(result,(1,1))


# In[70]:


#this is the scaled data.
#Inverse scaling is needed to see the original values.
#this can be done by using inverse_transform() function
tom_price=sc.inverse_transform(result)
print('predicted price tomorrow=%.2f'% tom_price)


# In[71]:


data = pd.read_csv('Reliance (1).csv')
data.head()


# In[72]:


data.dropna(axis = 0, inplace = True)


# In[73]:


del data['Adj Close']
data.shape


# In[74]:


meta = data.copy()
meta['Date'] = pd.to_datetime(meta['Date'], format='%Y-%m-%d')
meta['Year'] = meta['Date'].dt.year
meta.head()


# In[75]:


data.dropna(axis = 0, inplace = True)


# In[76]:


meta = data.copy()
meta['Date'] = pd.to_datetime(meta['Date'], format='%Y-%m-%d')
meta['Year'] = meta['Date'].dt.year
meta.head()


# In[77]:


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.scatter(x = meta['Year'], y = meta['Close'], color = 'green')
plt.xlabel('Period', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.show()


# In[78]:


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(meta['Year'],meta['Close'])
plt.xlabel('Period', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.show()


# In[79]:


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(meta['Close'], color = '#ba0459')
plt.xlabel('Day Count', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.show()


# In[80]:


plt.figure(figsize=(16,8))
plt.title('Volume Traded History')
plt.scatter(x = meta['Year'], y = meta['Volume'], color = '#0dffa6')
plt.xlabel('Year', fontsize=18)
plt.ylabel('Volume Traded', fontsize=18)
plt.show()


# In[81]:


plt.figure(figsize=(16,8))
plt.title('Volume Traded History')
plt.plot(meta['Volume'], color = '#bd0019')
plt.xlabel('Day Count', fontsize=18)
plt.ylabel('Volume Traded', fontsize=18)
plt.show()


# In[82]:


data.set_index('Date', inplace = True)
data.head()


# In[83]:


scaler = MinMaxScaler()
X = data[['Open', 'Low', 'High', 'Volume']].copy()
y = data['Close'].copy()


# In[84]:


X[['Open', 'Low', 'High', 'Volume']] = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))


# In[85]:


X_mat = X.values


# In[86]:


def load_data(X, seq_len, train_size=0.8):
    amount_of_features = X.shape[1]
    X_mat = X.values
    sequence_length = seq_len + 1
    datanew = []
    
    for index in range(len(X_mat) - sequence_length):
        datanew.append(X_mat[index: index + sequence_length])
    
    datanew = np.array(datanew)
    train_split = int(round(train_size * datanew.shape[0]))
    train_data = datanew[:train_split, :]
    
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1][:,-1]
    
    X_test = datanew[train_split:, :-1] 
    y_test = datanew[train_split:, -1][:,-1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  

    return X_train, y_train, X_test, y_test


# In[87]:


window = 22
X['close'] = y
X_train, y_train, X_test, y_test = load_data(X, window)
print(X_train.shape) 
print(y_train.shape) 
print(X_test.shape) 
print(y_test.shape)


# In[88]:


model = Sequential()
model.add(LSTM(128, input_shape= (window, 5), return_sequences = True))
model.add(Dropout(0.2))


# In[89]:


model.add(LSTM(128, input_shape = (window, 5), return_sequences=False))
model.add(Dropout(0.2))


# In[90]:


model.add(Dense(32))

model.add(Dense(1))


# In[91]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[92]:


model.fit(X_train, y_train, batch_size=1, validation_split = 0.1, epochs = 4)


# In[93]:


trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)


# In[94]:


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([y_test])


# In[96]:


trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f' % (trainScore))


# In[158]:


testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[145]:


plot_predicted = testPredict.copy()
plot_predicted = plot_predicted.reshape(242, 1)
plot_actual = testY.copy()
plot_actual = plot_actual.reshape(242, 1)


# In[146]:


plot_predicted_train = trainPredict.copy()
plot_predicted_train = plot_predicted_train.reshape(967, 1)
plot_actual_train = trainY.copy()
plot_actual_train = plot_actual_train.reshape(967, 1)


# In[147]:


plt.figure(figsize = (16,8))
plt.plot(pd.DataFrame(plot_predicted_train), label='Train Predicted')
plt.plot(pd.DataFrame(plot_actual_train), label='Train Actual')
plt.legend(loc='best')
plt.show()


# In[148]:


plt.figure(figsize = (16,8))
plt.plot(pd.DataFrame(plot_predicted), label='Test Predicted')
plt.plot(pd.DataFrame(plot_actual), label='Test Actual')
plt.legend(loc='best')
plt.show()


# In[ ]:





# In[ ]:




