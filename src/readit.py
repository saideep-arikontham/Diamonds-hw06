import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate

def read_diamonds():
    return pd.read_csv('data/diamonds.csv')

def convert_log(df,cols):
    for col in cols:
        df[col] = np.log(df[col])
    return df
    

def simple_train_test_split(X, y, test_size=.3):
    n_training_samples = int((1.0 - test_size) * X.shape[0])

    X_train = X[:n_training_samples,:]
    y_train = y[:n_training_samples]

    X_test = X[n_training_samples:,:]
    y_test = y[n_training_samples:]

    return X_train, X_test, y_train, y_test

def two_feature_linear_reg(df, k):
    X = df[['carat', k]]
    y = df['price']

    # Apply OneHotEncoder to the categorical column 'b'
    encoder = OneHotEncoder(sparse_output=False)
    encoded_b = encoder.fit_transform(X[[k]])

    # Concatenate the encoded column with the numerical column 'a'
    X = pd.concat([X.drop(k, axis=1), pd.DataFrame(encoded_b, columns=encoder.get_feature_names_out([k]))], axis=1)
    
    #Feature Matrix
    #print('X.shape') - (53940, 9)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict using the test set
    y_pred = model.predict(X)

    print(f'Explained variance score with {k} as second feature:',explained_variance_score(y,y_pred))
    


    
def five_fold_cross_val(df, k):
    
    X = df[['carat',k]]
    y = df['price']

    encoder = OneHotEncoder(sparse_output=False)
    encoded_b = encoder.fit_transform(X[[k]])

    # Concatenate the encoded column with the numerical column 'a'
    X = pd.concat([X.drop(k, axis=1), pd.DataFrame(encoded_b, columns=encoder.get_feature_names_out([k]))], axis=1)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    
    model = LinearRegression()
    
    res = cross_validate(model, X, y, cv=5, return_train_score=True)
    print(f'FOR {k}')
    print('-- Train scores:',res['train_score'])
    print('-- Test scores:',res['test_score'])
    print('-- mean train score',np.mean(res['train_score']))
    print('-- mean test score', np.mean(res['test_score']))
    print('-- train score standard deviation',np.std(res['train_score']))
    print('-- test score standard deviation',np.std(res['test_score']))
    print(f"-- Mean train score +/- std error : {np.mean(res['train_score']): 0.3f} +/- {np.std(res['train_score']): 0.3f}")
    print(f"-- Mean test score +/- std error : {np.mean(res['test_score']): 0.3f} +/- {np.std(res['test_score']): 0.3f}")
    print('\n\n\n')
