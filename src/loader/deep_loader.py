import pandas as pd
import numpy as np
import os
from src.embedder import WordVectorsEmbedder
from transformers import BertTokenizer, XLNetTokenizer

WORD_VECTORS = ["word2vec", "fasttext"]
class DeepLoader:
  def __init__(self, config):
    self.config = config

    if self.config["tokenizer_type"] == "bert":
      self.tokenizer = BertTokenizer.from_pretrained(self.config["model_name"])
    elif self.config["tokenizer_type"] == "xlnet":
      self.tokenizer = XLNetTokenizer.from_pretrained(self.config["model_name"])
    elif self.config["tokenizer_type"] in WORD_VECTORS:
      self.tokenizer = WordVectorsEmbedder(
        {
          **self.config["tokenizer_config"],
          "label_key": self.config["label_key"],
          "key_list": self.config["key_list"]
        }
      )
    
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

  def one_hot_encoding_based_on_labels(self, val):
    labels = self.config["labels"].split("_")
    arr = []
    for label in labels:
      if label == val:
        arr.append(1)
      else:
        arr.append(0)

  def one_hot_df(self, df: pd.DataFrame):
    arr = []
    for index, row in df.iterrows():
      arr.append(
        self.one_hot_encoding_based_on_labels(row[self.config["label_key"]])
      )
    return arr

  def __tokenize(self, sample):
    train = self.train
    dev = self.dev
    merged = self.merged_data
    test = self.test
    if sample:
      train = train.sample(5)
      dev = dev.sample(5)
      merged = merged.sample(5)
      if not test.empty:
        test = test.sample(5)
    
    train = train.reset_index(drop=True)
    dev = dev.reset_index(drop=True)
    merged = merged.reset_index(drop=True)
    test = test.reset_index(drop=True)

    y_train = np.array(self.one_hot_df(train[self.config["label_key"]]))
    y_dev = np.array(self.one_hot_df(dev[self.config["label_key"]]))
    y_test = np.array([])
    
    key_list = self.config["key_list"].split("_")
    x_train, x_dev, x_test = {}, {}, {}
    if self.config["tokenizer_type"] in WORD_VECTORS:
      x_train = self.tokenizer.df_to_vector(train)
      x_dev = self.tokenizer.df_to_vector(dev)
    else:
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
    
    if not self.test.empty:
      y_test = np.array(self.one_hot_df(test[self.config["label_key"]]))
      if self.config["tokenizer_type"] in WORD_VECTORS:
        x_test = self.tokenizer.df_to_vector(test)
      else:
        x_test = dict(
          self.tokenizer(
            list(self.test[key_list[0]]),
            list(self.test[key_list[1]]),
            **self.config["tokenizer_config"]
          )
        )
    
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

  def __call__(self, sample=False):
    self.__drop_null()
    return self.__tokenize(sample)



