import pandas as pd

df = pd.read_csv('data/kc_house_data.csv')
df.head()
df.describe().transpose()
df.isnull().sum()

# EDA
import matplotlib.pyplot as plt
import seaborn as sns

def plot(callplot: callable, *args, **kwargs):
    plt.figure(figsize=(12, 8))
    callplot(*args, **kwargs)


plot(sns.distplot, df['price'])
plot(sns.countplot, df['bedrooms'])
plot(sns.scatterplot, 'price', 'sqft_living', data=df)
plot(sns.boxplot, 'bedrooms', 'price', data=df)

# Geographical properties
plot(sns.scatterplot, x='price', y='long', data=df)
plot(sns.scatterplot, x='price', y='lat', data=df)
plot(sns.scatterplot, x='long', y='lat', data=df, hue='price')

# Sort df by descending price
df.sort_values('price', ascending=False).head(20)

# Length of 1%
perc_length = round(len(df)*0.01)

# Take the 1% top prices off
non_top_1_perc = df.sort_values('price', ascending=False).iloc[perc_length:]


plot(
    sns.scatterplot,
    x='long',
    y='lat',
    data=non_top_1_perc,
    hue='price',
    palette='RdYlGn',
    edgecolor=None,
    alpha=0.2
)
plot(sns.boxplot, x='waterfront', y='price', data=df)


# Feature data
df = df.drop('id', axis=1)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date: date.month)
df['year'] = df['date'].apply(lambda date: date.year)

plot(sns.boxplot, x='year', y='price', data=df)
plot(sns.boxplot, x='month', y='price', data=df)

df.groupby('month').mean()['price'].plot()
df.groupby('year').mean()['price'].plot()

df = df.drop('date', axis=1)

df['zipcode'].value_counts()
df = df.drop('zipcode', axis=1)

df['yr_renovated'].value_counts()
df['sqft_basement'].value_counts()


# Scaling and Train Test Split
X = df.drop('price', axis=1)
y = df['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=101
)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train.shape
X_test.shape


# Creating a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# Training the model
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test.values),
    batch_size=128,
    epochs=400
)

losses = pd.DataFrame(model.history.history)
losses.plot()


# Evaluation on test data
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score
)
import numpy as np

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

explained_variance_score(y_test, y_pred)

df['price'].mean()
df['price'].median()

def plot_errors():
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test, 'r')

plot_errors()

errors = y_test.values.reshape(6480, 1) - y_pred
sns.distplot(errors)


# Predicting on a brand new house
single_house = df.drop('price', axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1, 19))

single_pred = model.predict(single_house)
df.iloc[0]
