import pandas as pd
import os

class ShallowLoader:
  def __init__(self, config):
    self.config = config
    data_path = self.config["data_path"]
    self.train = pd.read_csv(os.path.join(data_path, "train.csv")).reset_index(drop=True)
    self.dev = pd.read_csv(os.path.join(data_path, "dev.csv")).reset_index(drop=True)
    self.merged_data = pd.concat([self.train, self.dev])
    print(f"train and dev dataset from {data_path} loaded!")
    
    self.test = pd.DataFrame([])
    if os.path.isfile(os.path.join(data_path, "test.csv")):
      self.test = pd.read_csv(os.path.join(data_path, "test.csv")).reset_index(drop=True)
      print(f"testdataset from {data_path} loaded!")

  def convert_label(self, val):
    labels = self.config["labels"].split("_")
    for i in range(len(labels)):
      if (labels[i] == val):
        return i
      else:
        continue

  def reverse_label(self, val):
    labels = self.config["labels"].split("_")
    return labels[val]
  
  def convert_label_df(self, df: pd.DataFrame):
    for index, row in df.iterrows():
      df.loc[index, self.config["label_key"]] = self.convert_label(row[self.config["label_key"]])

  def reverse_label_df(self, df):
    for index, row in df.iterrows():
      df.loc[index, self.config["label_key"]] = self.reverse_label(row[self.config["label_key"]])

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
    if not self.test.empty:
      self.test.dropna(inplace=True)
      if len(self.test.columns.tolist()) > 5:
        self.test.drop(columns=self.test.columns[0], axis=1, inplace=True)
    else:
      self.test = pd.DataFrame([])

  def __tokenize(self, sample):
    train = self.train
    dev = self.dev
    merged = self.merged_data
    test = self.test

    if sample:
      train = train.sample(20)
      dev = dev.sample(20)
      merged = merged.sample(20)
      if not test.empty:
        test = test.sample(20)

    if self.config["model_type"] != 'svm':
      self.convert_label_df(train)
      self.convert_label_df(dev)
      self.convert_label_df(test)
      self.convert_label_df(merged)
      
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
  
  def __call__(self, sample):
    self.__drop_null()
    return self.__tokenize(sample)