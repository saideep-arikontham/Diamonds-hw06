import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from readit import convert_log, read_diamonds

fig, axs = plt.subplots(1, 3, figsize=(20, 5)) 


df1 = convert_log(read_diamonds(),['carat'])
X = df1['carat'].values.reshape(-1, 1)
y = df1['price'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

sns.lineplot(data=df1, x='carat', y='price', label = 'actual data', ax = axs[0])
axs[0].legend()
axs[0].plot(X, y_pred, color='red', label = 'model prediction')
axs[0].legend()
axs[0].set_title('Linear regression curve: log(carat) vs price')
axs[0].text(0.95, 0.01, f'Explained variance score = {explained_variance_score(y,y_pred):.3f}',
        verticalalignment='bottom', horizontalalignment='right',
        transform=axs[0].transAxes,
        color='green', fontsize=12)
axs[0].set_xlabel('log(carat)')
print(f'The Explained variance score of the model for log(carat) to predict price is {explained_variance_score(y,y_pred)}')


df2 = convert_log(read_diamonds(),['price'])
X = df2['carat'].values.reshape(-1, 1)
y = df2['price'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

sns.lineplot(data=df2, x='carat', y='price', label = 'actual data', ax = axs[1])
axs[1].legend()
axs[1].plot(X, y_pred, color='red', label = 'model prediction')
axs[1].legend()
axs[1].set_title('Linear regression curve: carat vs log(price)')
axs[1].text(2.16, 0.01, f'Explained variance score = {explained_variance_score(y,y_pred):.3f}',
        verticalalignment='bottom', horizontalalignment='right',
        transform=axs[0].transAxes,
        color='green', fontsize=12)
axs[1].set_ylabel('log(price)')
print(f'The Explained variance score of the model for carat to predict log(price) is {explained_variance_score(y,y_pred)}')


df3 = convert_log(read_diamonds(),['carat','price'])
X = df3['carat'].values.reshape(-1, 1)
y = df3['price'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

sns.lineplot(data=df3, x='carat', y='price', label = 'actual data', ax = axs[2])
axs[2].legend()
axs[2].plot(X, y_pred, color='red', label = 'model prediction')
axs[2].legend()
axs[2].set_title('Linear regression curve: log(carat) vs log(price)')
axs[2].text(3.35, 0.01, f'Explained variance score = {explained_variance_score(y,y_pred):.3f}',
        verticalalignment='bottom', horizontalalignment='right',
        transform=axs[0].transAxes,
        color='green', fontsize=12)
axs[2].set_ylabel('log(price)')
axs[2].set_xlabel('log(carat)')
print(f'The Explained variance score of the model for carat to predict log(price) is {explained_variance_score(y,y_pred)}')


plt.savefig('figs/log_feature_prediction.png')
plt.show()
