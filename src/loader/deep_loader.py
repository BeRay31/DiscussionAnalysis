import pandas as pd
import numpy as np
import os
from src.embedder import WordVectorsEmbedder
from transformers import BertTokenizer, XLMRobertaTokenizer

WORD_VECTORS = ["word2vec", "fasttext"]
class DeepLoader:
  def __init__(self, config):
    self.config = config

    if self.config["tokenizer_type"] == "bert":
      self.tokenizer = BertTokenizer.from_pretrained(self.config["model_name"])
    elif self.config["tokenizer_type"] == "roberta":
      self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.config["model_name"])
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
    print(f"train and dev dataset from {data_path} loaded!")
    
    self.test = pd.DataFrame([])
    if os.path.isfile(os.path.join(data_path, "test.csv")):
      self.test = pd.read_csv(os.path.join(data_path, "test.csv"))
      print(f"testdataset from {data_path} loaded!")

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
    if not self.test.empty:
      self.test.dropna(inplace=True)
      if len(self.test.columns.tolist()) > 5:
        self.test.drop(columns=self.test.columns[0], axis=1, inplace=True)
    else:
      self.test = pd.DataFrame([])

  def one_hot_encoding_based_on_labels(self, val):
    labels = self.config["labels"].split("_")
    arr = []
    for label in labels:
      if label.lower() == val.lower():
        arr.append(1)
      else:
        arr.append(0)
    return arr

  def one_hot_df(self, items):
    arr = []
    for item in items:
      arr.append(
        self.one_hot_encoding_based_on_labels(item)
      )
    return arr

  def __tokenize(self, sample):
    train = self.train
    dev = self.dev
    merged = self.merged_data
    test = self.test
    if sample:
      train = train.sample(3)
      dev = dev.sample(3)
      merged = merged.sample(3)
      if not test.empty:
        test = test.sample(3)
    
    train = train.reset_index(drop=True)
    dev = dev.reset_index(drop=True)
    merged = merged.reset_index(drop=True)
    test = test.reset_index(drop=True)

    y_train = np.array(
      self.one_hot_df(
        train[self.config["label_key"]].tolist()
      )
    )
    y_dev = np.array(
      self.one_hot_df(
        dev[self.config["label_key"]].tolist()
      )
    )
    y_test = np.array([])
    
    key_list = self.config["key_list"].split("_")
    x_train, x_dev, x_test = {}, {}, {}
    if self.config["tokenizer_type"] in WORD_VECTORS:
      x_train = self.tokenizer.df_to_vector(train, False)
      x_dev = self.tokenizer.df_to_vector(dev, False)
    else:
      x_train = dict(
        self.tokenizer(
          list(train[key_list[0]]),
          list(train[key_list[1]]),
          **self.config["tokenizer_config"]
        )
      )
      x_dev = dict(
        self.tokenizer(
          list(dev[key_list[0]]),
          list(dev[key_list[1]]),
          **self.config["tokenizer_config"]
        )
      )
    
    if not self.test.empty:
      y_test = np.array(
        self.one_hot_df(
          test[self.config["label_key"]].tolist()
        )
      )
      if self.config["tokenizer_type"] in WORD_VECTORS:
        x_test = self.tokenizer.df_to_vector(test, False)
      else:
        x_test = dict(
          self.tokenizer(
            list(test[key_list[0]]),
            list(test[key_list[1]]),
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



