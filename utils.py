import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import RobustScaler
import numpy as np
import tensorflow as tf
import keras
from keras.losses import MeanAbsolutePercentageError

__all__ = ['load_raw_data', 'plot_timeseries', 'split_train_test', 'plot_data_split', 
'robust_scale', 'create_dataset', 'build_model', 'fit_to_model_get_history',
'plot_history', 'inverse_transform', 'plot_test_predict', 'get_mae']


def load_raw_data(file_path):
    raw_df = pd.read_csv(file_path, encoding='cp949',  parse_dates=['날짜'], index_col = '날짜')
    df_copy = raw_df.copy()
    return raw_df, df_copy

def plot_timeseries(x_axis, y_axis, x_label, y_label):
    plt.figure(figsize = (10,6))
    plt.plot(x_axis, y_axis, color='blue')
    plt.xlabel(x_label, {'fontsize': 12})
    plt.ylabel(y_label, {'fontsize': 12})
    return plt.show()

def split_train_test(dataframe_data, train_portion):
    train_size = round(int(len(dataframe_data) * train_portion))
    test_size = len(dataframe_data) - train_size
    train, test = dataframe_data.iloc[:train_size], dataframe_data.iloc[train_size:]
    print(f"Data split to train({train_portion}) : test({round(1-train_portion)}) = {len(train)} : {len(test)}")
    return train, test

def plot_data_split(train_dataset, test_dataset):
    plt.figure(figsize=(10,6))
    plt.plot(train_dataset.확진자)
    plt.plot(test_dataset.확진자)
    plt.xlabel("Time (day)")
    plt.ylabel("Daily Confirmed Cases")
    plt.legend(['Train set', 'Test set'], loc='upper right')
    return plt.show()

def robust_scale(train_data, test_data):
    f_transformer = RobustScaler()
    f_transformer =  f_transformer.fit(train_data.iloc[:, :-1].values)
    train_data.iloc[:, :-1] = f_transformer.transform(train_data.iloc[:, :-1].values)
    test_data.iloc[:, :-1] = f_transformer.transform(test_data.iloc[:, :-1].values)

    global cnt_transformer
    cnt_transformer = RobustScaler()
    cnt_transformer = cnt_transformer.fit(train_data[['확진자']])
    train_data['확진자'] = cnt_transformer.transform(train_data[['확진자']])
    test_data['확진자'] = cnt_transformer.transform(test_data[['확진자']])
    return train_data, test_data

def create_dataset(X, y, time_steps=1):
    print(f"Reshaping to [{X.shape[0]}, {time_steps}, {X.shape[-1]}]")
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_model(X_train, units=128):
    tf.random.set_seed(42)
    model = keras.Sequential()
    model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(
        units=units,
        input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    )
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def fit_to_model_get_history(model, X_train, y_train, ep=100, b_size=32, val_split=0.1, shf=False):
    history = model.fit(X_train, y_train, epochs=ep, batch_size=b_size, validation_split=val_split, shuffle=shf)
    return history

def plot_history(history):
    plt.plot(history.history['loss'], label=['train'])
    plt.plot(history.history['val_loss'], label=['validation'])
    plt.legend()
    return plt.show()

def inverse_transform(y_train, y_test, y_pred):
    y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))
    y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
    y_pred_inv = cnt_transformer.inverse_transform(y_pred)
    return y_train_inv, y_test_inv, y_pred_inv

def plot_test_predict(y_test_inv, y_pred_inv):
    plt.plot(y_test_inv.flatten(), marker='.', label='true')
    plt.plot(y_pred_inv.flatten(),'r', label='prediction', marker='x')
    plt.legend()
    return plt.show()

def get_mae(y_test,y_pred):
    mae = MeanAbsolutePercentageError()
    return mae(y_test, y_pred).numpy()
    

