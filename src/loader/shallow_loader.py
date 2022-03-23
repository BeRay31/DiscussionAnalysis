import pandas as pd
import os

class ShallowLoader:
  def __init__(self, config):
    self.config = config
    data_path = self.config["data_path"]
    self.train = pd.read_csv(os.path.join(data_path, "train.csv")).reset_index(drop=True)
    self.dev = pd.read_csv(os.path.join(data_path, "dev.csv")).reset_index(drop=True)
    if os.path.isfile(self.config["test_data_path"]):
      self.test = pd.read_csv(self.config["test_data_path"]).reset_index(drop=True)
    self.merged_data = pd.concat([self.train, self.dev])
    print(f"train and dev dataset from {data_path} loaded!")

  def __drop_null(self):
    self.train.dropna(inplace=True)
    self.dev.dropna(inplace=True)
    self.merged_data.dropna(inplace=True)
    if len(self.train.columns.tolist()) > 5:
      self.train.drop(columns=self.train.columns[0], axis=1, inplace=True)
    if len(self.dev.columns.tolist()) > 5:
      self.dev.drop(columns=self.dev.columns[0], axis=1, inplace=True)
    if len(self.merged_data.columns.tolist()) > 5:
      self.merged_data.drop(columns=self.merged_data.columns[0], axis=1, inplace=True)
    if os.path.isfile(self.config["test_data_path"]):
      self.test.dropna(inplace=True)
      if len(self.test.columns.tolist()) > 5:
        self.test.drop(columns=self.test.columns[0], axis=1, inplace=True)
    else:
      self.test = pd.DataFrame([])

  def __tokenize(self):
    train = self.train
    dev = self.dev
    merged = self.merged_data
    test = self.test
      
    x_train = train.drop([self.config["label_key"]], axis=1)
    y_train = train[self.config["label_key"]]
    x_dev = dev.drop([self.config["label_key"]], axis=1)
    y_dev = dev[self.config["label_key"]]
    x_test = pd.DataFrame([])
    y_test = pd.DataFrame([])
    if not self.test.empty:
      x_test = test.drop([self.config["label_key"]], axis=1)
      y_test = test[self.config["label_key"]]

    res = {
      "x_train": x_train,
      "x_dev": x_dev,
      "x_test": x_test,
      "y_train": y_train,
      "y_dev": y_dev,
      "y_test": y_test,
      "merged": merged,
      "train": train,
      "dev": dev,
      "test": test
    }

    return res
  
  def __call__(self):
    self.__drop_null()
    return self.__tokenize()