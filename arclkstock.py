import cv2
import pandas as pd
import seaborn as sns
import numpy as np
import random, os, glob
import matplotlib.pyplot as plt
import datetime as dt
#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Model metric
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report,mean_squared_error
import tensorflow as tf
#Model set
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Dense,Dropout,LSTM


#Tensorflow filter warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#check dataframe
def check_df(dataframe,head=5):
    print("----------------Shape----------------\n")
    print(dataframe.shape)
    print("----------------Types----------------\n")
    print(dataframe.dtypes)
    print("----------------Tail----------------\n")
    print(dataframe.tail(head))
    print("----------------Null?----------------\n")
    print(dataframe.isnull().sum())
    print("----------------Quantiles----------------\n")
    print(dataframe.quantile([0,0.05,0.50,0.95,0.99,1]).T)

df = pd.read_csv("ARCLK.csv")

check_df(df,5)

df = df.dropna()
print("İs null ? \n => ",df.isnull().sum())
stock_data = df[["Date","Close"]]

stock_data.index = stock_data["Date"]

stock_data.drop("Date",axis=1,inplace=True)

result_df = stock_data.copy()

#df to numpy array
stock_data = stock_data.values
#Changing dtype to float32
stock_data = stock_data.astype('float32')

#Split data to train & test

def split_data(dataframe,test_size):
    pos = int(round(len(dataframe) * (1-test_size)))
    train = dataframe[:pos]
    test = dataframe[pos:]
    return train,test,pos

train,test,pos = split_data(stock_data,0.20)

print(train.shape,test.shape)


#Scaling
'''Değerleri 0 ve 1 aralığına sıkıştırıyoruz ki gradyan ezilmesi olmasın ve işlem uzunluğu kısalsın'''
scaler_train = MinMaxScaler(feature_range=(0,1))
train = scaler_train.fit_transform(train)
scaler_test = MinMaxScaler(feature_range=(0,1))
test = scaler_test.fit_transform(test)

def create_features(data,lookback):
    X,Y = [], []
    for i in range(lookback,len(data)):
        X.append(data[i-lookback:i,0])
        Y.append(data[i,0])
    
    return np.array(X),np.array(Y)

lookback = 20

X_train,y_train = create_features(train,lookback)

X_test,y_test = create_features(test,lookback)

#Preprocessing

X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

'''Model setup'''

model = Sequential()    
model.add(LSTM(
    units=50,   
    activation='relu',  
    input_shape=(X_train.shape[1],lookback)
    ))

model.add(Dropout(0,2))

model.add(Dense(1)) 

print(model.summary())

#Optimization & Accuracy metrics
model.compile(loss='mean_squared_error',optimizer='adam')

#callback func for overfitting
callbacks = [EarlyStopping(monitor='val_loss',patience=3,verbose=1,mode='min'),
             ModelCheckpoint(filepath='mymodel.h5',monitor='val_loss',mode='min',
             save_best_only=True,save_weights_only=False,verbose=1)
             ]

#Train model
history = model.fit(x=X_train,
                    y=y_train,
                    epochs=100, #Optimizasyon turu
                    batch_size=20,  #Verigrubunun boyutu
                    validation_data=(X_test,y_test), #test setimizi veriyoruz
                    callbacks=callbacks,    #aşırı öğrenmenin önüne geçmek
                    shuffle=False
                    )



plt.figure(figsize=(20,5))
plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss',fontsize=16)
plt.show()



loss = model.evaluate(X_test,y_test,batch_size=20)
print("\n Test loss:%.1f%%",(100.0*loss))

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#Unscale
train_predict = scaler_train.inverse_transform(train_predict)
test_predict = scaler_test.inverse_transform(test_predict)


y_train =scaler_train.inverse_transform(y_train)
y_test =scaler_test.inverse_transform(y_test)

#RMSE
train_rmse = np.sqrt(mean_squared_error(y_train,train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test,test_predict))

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

train_prediction_df = result_df[lookback:pos]
train_prediction_df["Predicted"] = train_predict
print(train_prediction_df.head())

test_prediction_df = result_df[pos+lookback:]
test_prediction_df["Predicted"] = test_predict
print(test_prediction_df.head())


plt.figure(figsize=(15,5))
plt.plot(result_df,label='Real Values')
plt.plot(train_prediction_df['Predicted'],color='blue',label="Train Predicted")
plt.plot(test_prediction_df['Predicted'],color='red',label='Test Predicted')
plt.xlabel('Time')
plt.ylabel('Stock Values')
plt.legend()
plt.show()

print(test_prediction_df.head(100))

