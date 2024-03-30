import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from readit import convert_log, read_diamonds, simple_train_test_split



df = convert_log(read_diamonds(),['carat','price'])

X = df['carat'].values.reshape(-1, 1)
y = df['price'].values
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Using sklearn train_test_split, we get Explained Variance Score as: ',explained_variance_score(y_test,y_pred))


axs[0].plot(X_test,y_pred, color = 'black', linewidth = '10', label = f'sklearn train_test_split (score = {explained_variance_score(y_test,y_pred):.3f})')
axs[0].legend()
a = X_test.ravel()

X_train, X_test, y_train, y_test = simple_train_test_split(X, y, test_size=.3)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('\nUsing user-defined simple_train_test_split, we get Explained Variance Score as: ',explained_variance_score(y_test,y_pred))

axs[0].plot(X_test,y_pred, color = 'orange', linewidth = '5', label = f'simple_train_test_split (score = {explained_variance_score(y_test,y_pred):.3f})')
axs[0].legend()
b= X_test.ravel()

axs[1].boxplot([a,b])
axs[1].set_xticklabels(['X_test: sklearn train_test_split','X_test: simple_train_test_split'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=False, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('\nWhen we pass the parameter "shuffle = False" to sklearn train_test_split, we get Explained Variance Score as: ',explained_variance_score(y_test,y_pred))

print("\nTherefore, Shuffle is the main difference between two functions")

plt.savefig('figs/linear_reg_plots.png')
plt.show()
