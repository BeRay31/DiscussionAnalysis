from pandas import DataFrame
from src.util import get_config
from src.loader import DeepLoader
from src.classifier import DeepClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import pandas as pd
default_config = {
  'master': {
    'model_name': 'cardiffnlp/twitter-xlm-roberta-base',
    'labels': 'Pro_Neutral_Contra',
    'model_weight_path': '/raid/data/m13518136/res/deep/final_label/xlmroberta-normalros-0.0.0/models/model_0.h5',
    'label_key': 'Label'
  },
  'master_binary': {
    'model_name': 'cardiffnlp/twitter-xlm-roberta-base',
    'labels': 'Uncorrelated_Correlated',
    'model_weight_path': '/raid/data/m13518136/res/deep/BinaryPipeline/xlmroberta-normalros-binary-0.0.1/models/model_0.h5',
    'label_key': 'Correlation'
  },
  'model_config': {
    'type': 'roberta',
    'recurrent_layer': 'bilstm',
    'recurrent_unit': 64,
    'recurrent_dropout': 0.4,
  },
  'loader_config': {
    'tokenizer_type': 'roberta',
    'tokenizer_config': {
      'max_length': 512,
      'padding': 'max_length',
      'return_tensors': 'tf',
      'truncation': True
    }
  }
}


df_config = {
  'key': "Tweet_Comment"
}

sample_tweet = ["test", "test", "test"]
sample_retweet = ["test", "test", "test"]

class Wrapper:
  def __init__(self, config_path = "", config = default_config):
    if (config_path):
      self.config = get_config(config_path)
    else:
      self.config = config
    
    self.loader = DeepLoader(
      {**self.config["master"], **self.config["loader_config"]}
    )
    self.model = DeepClassifier(
      {**self.config["master"], **self.config["model_config"]}
    )
    self.model_binary = DeepClassifier(
      {**self.config["master_binary"], **self.config["model_config"]}
    )
    self.sample_data = self.loader.tokenize(sample_tweet, sample_retweet)
    self.model(self.sample_data)
    self.model_binary(self.sample_data)

    self.model.load_weights(self.config['master']['model_weight_path'])
    self.model_binary.load_weights(self.config['master_binary']['model_weight_path'])
  
  def get_label(self, arr, masterConfig):
    labels = masterConfig["labels"].split("_")
    idx = np.argmax(arr)
    return labels[idx]

  def predict(self, tweet, retweet):
    input = self.loader.tokenize(tweet, retweet)
    pred = self.model.predict(input)
    return [self.get_label(item) for item in pred]
  
  def predict_batch_retweet(self, tweet, retweets):
    tweets = [tweet for i in range(len(retweets))]
    input = self.loader.tokenize(tweets, retweets)
    pred = self.model.predict(input)
    return [self.get_label(item) for item in pred]
  
  def predict_2Pipeline_df(self, df: DataFrame):
    key_list = self.config["key_list"].split("_")
    input = self.loader.tokenize(df[key_list[0]], df[key_list[1]])

    relation = self.model_binary.predict(input)
    relation_prediction = [self.get_label(arr, self.config["master_binary"]) for arr in relation]
    relation_prediction = pd.DataFrame(relation_prediction, columns=["Binary Correlation"])
    pred = pd.concat([df, relation_prediction], axis = 1)

    relation_conf_matrix = confusion_matrix(
      y_true=pred[self.config["master_binary"]["label_key"]],
      y_pred=pred["Binary Correlation"],
      labels=self.config["master_binary"]["labels"].split("_")
    )
    relation_classification_report = classification_report(
      y_true=pred[self.config["master_binary"]["label_key"]],
      y_pred=pred["Binary Correlation"],
      labels=self.config["master_binary"]["labels"].split("_")
    )
    relation_accuracy_score = accuracy_score(
      y_true=pred[self.config["master_binary"]["label_key"]],
      y_pred=pred["Binary Correlation"]
    )
    relation_f1_score = f1_score(
      y_true=pred[self.config["master_binary"]["label_key"]],
      y_pred=pred["Binary Correlation"],
      labels=self.config["master_binary"]["labels"].split("_")
      average='weighted'
    )

    # filter uncorrelated pred
