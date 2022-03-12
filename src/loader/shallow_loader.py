import pandas as pd
import os

class ShallowLoader:
  def __init__(self, config):
    self.config = config
    data_path = self.config["data_path"]
    self.train = pd.read_csv(os.path.join(data_path, "train.csv")).reset_index(drop=True)
    self.test = pd.read_csv(os.path.join(data_path, "test.csv")).reset_index(drop=True)
    self.merged_data = pd.concat([self.train, self.test])
    print(f"train and test dataset from {data_path} loaded!")

  def __drop_null(self):
    self.train.dropna(inplace=True)
    self.test.dropna(inplace=True)
    self.merged_data.dropna(inplace=True)
    if len(self.train.columns.tolist()) > 5:
      self.train.drop(columns=self.train.columns[0], axis=1, inplace=True)
    if len(self.test.columns.tolist()) > 5:
      self.test.drop(columns=self.test.columns[0], axis=1, inplace=True)
    if len(self.merged_data.columns.tolist()) > 5:
      self.merged_data.drop(columns=self.merged_data.columns[0], axis=1, inplace=True)
  
  def __tokenize(self, sample=False):
    if sample:
      train = self.train.sample(5)
      test = self.test.sample(5)
      merged = self.merged_data.sample(5)
    else:
      train = self.train
      test = self.test
      merged = self.merged_data
      
    x_train = train.drop(["Label"], axis=1)
    y_train = train["Label"]
    x_test = test.drop(["Label"], axis=1)
    y_test = test["Label"]

    res = {
      "x_train": x_train,
      "x_test": x_test,
      "y_train": y_train,
      "y_test": y_test,
      "train": train,
      "merged": merged,
      "test": test
    }

    return res
  
  def __call__(self, sample = False):
    self.__drop_null()
    return self.__tokenize(sample)