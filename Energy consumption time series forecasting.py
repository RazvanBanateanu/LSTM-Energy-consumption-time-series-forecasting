from time import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from SeriesToSupervised import series_to_supervised

dataset = pd.read_csv('WT_cladiri.csv')
dataset_time = np.array(dataset['Timp'])

print('Dataset time shape: ',dataset_time.shape)

# manipulate data
year= []
month = []
day = []
hour = []

for i in range(len(dataset)):
    a = dataset['Timp'][i]
    my_date=datetime.strptime(a, '%d.%m.%Y %H:%M')
    year.append(my_date.year)
    month.append(my_date.month)
    day.append(my_date.weekday())
    hour.append(my_date.hour)

dataset['An']= year
dataset['Luna']= month
dataset['Zi']= day
dataset['Ora']= hour
del dataset['Timp']
swi = dataset.pop("HVAC")
dataset['HVAC']= swi

dataset = dataset.loc[:,('HVAC', 'Temperatura','An','Luna','Zi','Ora')]

print('Dataset shape: ',dataset.shape)

data = dataset.astype(float)
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

print('Data shape(scaler): ',data.shape)

# size = 0.9998
# size = 0.994
size = 1
train_test = int(len(data))*size

data_train_test = data[:int(train_test)]

print('data_train_test shape: ', data_train_test.shape)

data_predict = data[int(train_test):]

print('data_predict shape: ', data_predict.shape)


dataset_time_train_test = dataset_time[:int(train_test)]

print('dataset_time_train_test shape: ', dataset_time_train_test.shape)

dataset_time_predict = dataset_time[int(train_test):]

print('dataset_time_predict shape: ', dataset_time_predict.shape)

# convert time series into supervised learning problem
lookback = 1
lookforward = 1
reframed = series_to_supervised(data_train_test, lookback, lookforward)


reframed.drop(reframed.columns[[7, 8, 9, 10, 11]], axis=1, inplace=True)

print('Reframed shape: ', reframed.shape)



#################

predict = series_to_supervised(data_predict, lookback, lookforward)
predict.drop(predict.columns[[7, 8, 9, 10, 11]], axis=1, inplace=True)
print("predict  ", predict.shape)
data_predict = predict.values
data_predict_X = data_predict[: , :-1]

print('data_predict_X shape: ', data_predict_X.shape)

data_predict_X = data_predict_X.reshape((data_predict_X.shape[0], 1, data_predict_X.shape[1]))

print('data_predict_X_reshaped shape: ', data_predict_X.shape)

data_predict_Y = data_predict[:, -1]
data_predict_Y = data_predict_Y.reshape((len(data_predict_Y), 1))

print('data_predict_Y shape: ', data_predict_Y.shape)

######################

values = reframed.values
print('values shape: ', values.shape)

size = 0.8
training_size=int(len(data_train_test)*size)

train = values[:training_size]
print('train shape: ', train.shape)

test = values[training_size:]
print('test shape: ', test.shape)

dataset_time_train = dataset_time_train_test[:training_size]
print('dataset_time_train shape: ', dataset_time_train.shape)

dataset_time_test = dataset_time_train_test[training_size:]
print('dataset_time_test shape: ', dataset_time_test.shape)

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]


print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
print('train_X shape: ', train_X.shape)

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print('test_X shape: ', test_X.shape)

# model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(LSTM(50, return_sequences=True))
# model.add(LSTM(50))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

# #fit network
# history = model.fit(train_X, train_y, epochs=33, batch_size=48, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# model.save('LicentaV4_05.h5')

# model = load_model('LicentaV4_04.h5')


# train_predict=model.predict(train_X)
# print('train_predict: ', train_predict.shape)

# train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))

# inv_yhat = np.concatenate((train_predict, train_X[:, 1:]), axis=1)
# print('inv_yhat: ', inv_yhat.shape)

# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]

# train_y = train_y.reshape((len(train_y), 1))
# inv_y = np.concatenate((train_y, train_X[:, 1:]), axis=1)
# print('inv_y: ', inv_y.shape)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]





# print(inv_yhat[:20])

# print(inv_y[:20])

# rmse2 = math.sqrt(mean_squared_error(inv_y, inv_yhat))
# print("R2:", r2_score(inv_y, inv_yhat))
# print('Test RMSE2: %.3f' % rmse2)

##########################################################################

# train_predict=model.predict(test_X)
# print('train_predict: ', train_predict.shape)

# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# inv_yhat = np.concatenate((train_predict, test_X[:, 1:]), axis=1)
# print('inv_yhat: ', inv_yhat.shape)

# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]

# test_y = test_y.reshape((len(test_y), 1))
# inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
# print('inv_y: ', inv_y.shape)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]


# print(inv_yhat[:20])

# print(inv_y[:20])

# rmse2 = math.sqrt(mean_squared_error(inv_y, inv_yhat))
# print("R2:", r2_score(inv_y, inv_yhat))
# print('Test RMSE2: %.3f' % rmse2)

# plt.plot(dataset_time_test[:-1],inv_yhat, label = 'predict')
# plt.plot(dataset_time_test[:-1],inv_y, label = 'real')
# plt.show()


#######################################################

# train_predict=model.predict(data_predict_X)

# print('train_predict: ', train_predict.shape)

# data_predict_X = data_predict_X.reshape((data_predict_X.shape[0], data_predict_X.shape[2]))

# inv_yhat = np.concatenate((train_predict, data_predict_X[:, 1:]), axis=1)
# print('inv_yhat: ', inv_yhat.shape)

# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]


# inv_y = np.concatenate((data_predict_Y, data_predict_X[:, 1:]), axis=1)
# print('inv_y: ', inv_y.shape)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]


# print(inv_yhat)

# print(inv_y)

# rmse2 = math.sqrt(mean_squared_error(inv_y, inv_yhat))
# print("R2:", r2_score(inv_y, inv_yhat))
# print('Test RMSE2: %.3f' % rmse2)

# plt.plot(dataset_time_predict[:-1],inv_yhat, label = 'predict')
# plt.plot(dataset_time_predict[:-1],inv_y, label = 'real')
# plt.show()


######################################################################

# my_X= [[0.21077878, 0.36919692, 1, 0.36363636, 0.83333333, 0.60869565],
#        [0.13544018, 0.35709571, 1, 0.36363636, 0.83333333, 0.65217391],
#        [0.05953725, 0.3379538,  1, 0.36363636, 0.83333333, 0.69565217]]


# my_X= [[0.13544018, 0.35709571, 1, 0.36363636, 0.83333333, 0.65217391]]

# my_X = np.array(my_X)
# my_X = my_X.reshape((1, 1 , 6))

# train_predict=model.predict(my_X)
# print('train_predict: ', train_predict.shape)

# my_X = my_X.reshape((1, data_predict_X.shape[2]))

# inv_yhat = np.concatenate((train_predict, my_X[:, 1:]), axis=1)
# print('inv_yhat: ', inv_yhat.shape)

# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]

# test_y = test_y.reshape((len(test_y), 1))
# inv_y = np.concatenate((test_y, my_X[:, 1:]), axis=1)
# print('inv_y: ', inv_y.shape)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]


# print(inv_yhat[:20])

# print(inv_y[:20])

# rmse2 = math.sqrt(mean_squared_error(inv_y, inv_yhat))
# print("R2:", r2_score(inv_y, inv_yhat))
# print('Test RMSE2: %.3f' % rmse2)

# print(train_predict)
# print(test_y)
