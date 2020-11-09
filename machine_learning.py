import math 
import quandl
import pandas as pd 
import numpy as np 
from sklearn import preprocessing,svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from matplotlib import style
import datetime
style.use('ggplot')



df=pd.read_csv('/home/sina/Downloads/EURUSD5 (2).csv',header=0,index_col='date',parse_dates=True)
# print(df)

df= df [['open','high','low','close','volume',]]

df['HL_PCT']=(df['high']-df['close'])/df['close']*100.0

df['PCT_change']=(df['close']-df['open'])/df['open']*100.0  

df = df[['close','HL_PCT','PCT_change','volume']]

forecast_col='close'
df.fillna(-9999,inplace=True)
forecast_out=int(math.ceil(0.01*len(df)))
# print(forecast_out)
df['label']=df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))


X=preprocessing.scale(X)
X=X[:-forecast_out]
# X_lately=X[-forecast_out:]

X_lately=X[-10:]

df.dropna(inplace=True)

y = np.array(df['label'])

df.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
clf=LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
forecast_set=clf.predict(X_lately)
# print(forecast_set, accuracy,forecast_out)
#print(forecast_set,accuracy)
df['forecast']=np.nan
# print(forecast_set[-10:])
# print('-----------------------',forecast_out)
# print(accuracy)
last_date=df.iloc[-1].name
last_unix= last_date.timestamp()
one_day=86400
next_unix=last_unix + one_day
for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day 
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)] +[i]
print(df.tail())
# df['close'].plot()
# df['forecast'].plot()
# plt.legend(loc=4)
# plt.xlabel('date')
# plt.ylabel('price')
# plt.show()