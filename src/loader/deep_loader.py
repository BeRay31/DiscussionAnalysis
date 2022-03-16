import pandas as pd
import numpy as np
import os

from transformers import BertTokenizer, XLNetTokenizer

class DeepLoader:
  def __init__(self, config):
    self.config = config

    if self.config["tokenizer_type"] == "bert":
      self.tokenizer = BertTokenizer.from_pretrained(self.config["model_name"])
    elif self.config["tokenizer_type"] == "xlnet":
      self.tokenizer = XLNetTokenizer.from_pretrained(self.config["model_name"])
    
    data_path = self.config["data_path"]
    self.train = pd.read_csv(os.path.join(data_path, "train.csv")).reset_index(drop=True)
    self.dev = pd.read_csv(os.path.join(data_path, "dev.csv")).reset_index(drop=True)
    self.merged_data = pd.concat([self.train, self.dev])
    if os.path.isfile(self.config["test_data_path"]):
      self.test = pd.read_csv(self.config["test_data_path"])
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
    # handle test data
    if os.path.isfile(self.config["test_data_path"]):
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

    key_list = self.config["key_list"].split("_")
    x_train, x_dev, x_test = {}, {}, {}
    x_train = dict(
      self.tokenizer(
        list(self.train[key_list[0]]),
        list(self.train[key_list[1]]),
        **self.config["tokenizer_config"]
      )
    )
    x_dev = dict(
      self.tokenizer(
        list(self.dev[key_list[0]]),
        list(self.dev[key_list[1]]),
        **self.config["tokenizer_config"]
      )
    )
    y_train = np.array(train[self.config["label_key"]])
    y_dev = np.array(dev[self.config["label_key"]])
    y_test = np.array([])
    if not self.test.empty:
      x_test = dict(
        self.tokenizer(
          list(self.test[key_list[0]]),
          list(self.test[key_list[1]]),
          **self.config["tokenizer_config"]
        )
      )
      y_test = np.array(test[self.config["label_key"]])
    
    res = {
      "x_train": x_train,
      "x_dev": x_dev,
      "x_test": x_test,
      "y_train": y_train,
      "y_dev": y_dev,
      "y_test": y_test,
      "train": train,
      "merged": merged,
      "dev": dev
    }

    return res

  def __call__(self):
    self.__drop_null()
    return self.__tokenize()


