import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
  def __init__(self, data_X, data_y, start_idx):
      self.data_X = data_X
      self.data_y = data_y
      self.start_idx = start_idx

  def __len__(self):
     return len(self.data_X)
  
  def __getitem__(self, index):
    if torch.is_tensor(index):
      index = index.tolist()
    return self.data_X[index], self.data_y[index]

def data_cleaning(data):
    df = pd.read_csv(data["sp_500_path"])
    
    df.index = df.Date

    df.drop('Date', axis = 1, inplace=True)
    columns = df.columns
    columns = columns[1:]
    df = df[columns]
    columns = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    df.columns = columns

    for i in columns:
      df[i] = df[i].replace(',', '', regex=True)
      df[i] = df[i].astype('float')

    df.sort_values(by='Date', key=pd.to_datetime, inplace=True)

    df['Return'] = df.Close - df.Open 
    df['DailyVolatility'] = df.High - df.Low
    ewdf = df.ewm(halflife=10).mean()
    vewdf = df.ewm(halflife=10).var()

    df['ScaledVolatility'] = (df.DailyVolatility - ewdf.DailyVolatility) / (vewdf.DailyVolatility ** 0.5)
    df['ScaledReturn'] = (df.Return - ewdf.Return) / (vewdf.Return ** 0.5)
    df['ScaledVolume'] = (df.Volume - ewdf.Volume) / (vewdf.Volume ** 0.5)

    df.dropna(inplace=True)

    return df

def data_preprocessing(x, y, window_size):

  num_to_unpack = math.floor(x.shape[0] / window_size)
  start_idx = x.shape[0] - num_to_unpack * window_size
  x = x[start_idx:]
  y = y[start_idx:]

  x = np.expand_dims(x, axis = 1)
  x = np.split(x, x.shape[0]/window_size, axis = 0)
  x = np.concatenate(x, axis = 1)
  x = np.transpose(x, axes = (1, 0, 2))
  y = y[::window_size]

  x = x.astype(np.float32) #before transform into torch tensor
  y = y.astype(np.float32)

  x = torch.tensor(x)
  y = torch.tensor(y)

  return (x, y, start_idx)

def data_preparation(configs):
   
  clean_data = data_cleaning(configs["data"])

  train_test_split = configs["train"]["train_test_split"]
  train_split = int(math.floor(clean_data.shape[0] * train_test_split[0]))
  val_split = int(math.floor(clean_data.shape[0] * (train_test_split[0] + train_test_split[1])))

  prepared_train_data = clean_data[:train_split]
  prepared_val_data = clean_data[train_split:val_split]
  prepared_test_data = clean_data[val_split:]

  time_stamps = clean_data.index
  time_stamps_train = time_stamps[:train_split]
  time_stamps_val = time_stamps[train_split:val_split]
  time_stamps_test = time_stamps[val_split:]

  X_train = prepared_train_data[:(prepared_train_data.shape[0] - configs["model"]["window_size"])][['ScaledVolatility', 'ScaledReturn', 'ScaledVolume']].values
  Y_train = prepared_train_data[configs["model"]["window_size"]:]['ScaledReturn'].values
  time_stamps_train = time_stamps_train[:(time_stamps_train.shape[0] - configs["model"]["window_size"])]
  data_train_x, data_train_y, start_idx_train = data_preprocessing(X_train, Y_train, configs["model"]["window_size"])
  train_data = StockDataset(data_train_x, data_train_y, start_idx_train)
  data_train_iter = DataLoader(train_data, batch_size=configs["train"]["batch_size"], shuffle=False)

  X_val = prepared_val_data[:(prepared_val_data.shape[0] - configs["model"]["window_size"])][['ScaledVolatility', 'ScaledReturn', 'ScaledVolume']].values
  Y_val = prepared_val_data[configs["model"]["window_size"]:]['ScaledReturn'].values
  time_stamps_val = time_stamps_val[:(time_stamps_val.shape[0] - configs["model"]["window_size"])]
  data_val_x, data_val_y, start_idx_val = data_preprocessing(X_val, Y_val, configs["model"]["window_size"])
  val_data = StockDataset(data_val_x, data_val_y, start_idx_val)
  data_val_iter = DataLoader(val_data, batch_size=configs["train"]["batch_size"], shuffle=False)

  X_test = prepared_test_data[:(prepared_test_data.shape[0] - configs["model"]["window_size"])][['ScaledVolatility', 'ScaledReturn', 'ScaledVolume']].values
  Y_test = prepared_test_data[configs["model"]["window_size"]:]['ScaledReturn'].values
  time_stamps_test = time_stamps_test[:(time_stamps_test.shape[0] - configs["model"]["window_size"])]
  data_test_x, data_test_y, start_idx_test = for_testing_time(X_test, Y_test, configs["model"]["window_size"])
  test_data = StockDataset(data_test_x, data_test_y, start_idx_test)
  data_test_iter = DataLoader(test_data, batch_size=1, shuffle=False)

  return ((data_train_iter, data_val_iter, data_test_iter), (time_stamps_train, time_stamps_val, time_stamps_test))


def for_testing_time(x, y, window_size):
  
  x = np.lib.stride_tricks.sliding_window_view(x, (window_size, 3))[:-1]
  y = y[window_size:]

  x = x.astype(np.float32) #before transform into torch tensor
  y = y.astype(np.float32)

  x = torch.tensor(x)
  y = torch.tensor(y)

  start_idx = 0

  return (x, y, start_idx)