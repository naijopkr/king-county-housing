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
