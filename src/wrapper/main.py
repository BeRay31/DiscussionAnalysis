import os
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
    'label_key': 'Label',
    'final_label': "Uncorrelated_Pro_Neutral_Contra"
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
  },
  'two_pipeline': True,
  'key_list': "Tweet_Comment",
  'save_path': "D:\\WorkBench\\TA NLP\\res"
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
    self.sample_data = self.loader.tokenize(sample_tweet, sample_retweet)
    self.model(self.sample_data)
    self.model.load_weights(self.config['master']['model_weight_path'])

    if (self.config["two_pipeline"]) {
      self.model_binary = DeepClassifier(
        {**self.config["master_binary"], **self.config["model_config"]}
      )
      self.model_binary(self.sample_data)

      self.model_binary.load_weights(self.config['master_binary']['model_weight_path'])
    }
  
  def get_label(self, arr, masterConfig):
    labels = masterConfig["labels"].split("_")
    idx = np.argmax(arr)
    return labels[idx]

  def predict(self, tweet, retweet):
    input = self.loader.tokenize(tweet, retweet)
    pred = self.model.predict(input, verbose=1)
    return [self.get_label(item) for item in pred]
  
  def predict_batch_retweet(self, tweet, retweets):
    tweets = [tweet for i in range(len(retweets))]
    input = self.loader.tokenize(tweets, retweets)
    pred = self.model.predict(input, verbose=1)
    return [self.get_label(item) for item in pred]
  
  def predict_2Pipeline_df(self, df: DataFrame):
    key_list = self.config["key_list"].split("_")
    binary_input = self.loader.tokenize(df[key_list[0]], df[key_list[1]])

    relation = self.model_binary.predict(binary_input, verbose=1)
    relation_prediction = [self.get_label(arr, self.config["master_binary"]) for arr in relation]
    relation_prediction = pd.DataFrame(relation_prediction, columns=["Binary Correlation"])
    pred = pd.concat([df.reset_index(drop=True), relation_prediction.reset_index(drop=True)], axis=1)
    pred.to_csv(os.path.join(self.config["save_path"], "correlation.csv"))

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
      labels=self.config["master_binary"]["labels"].split("_"),
      average='weighted'
    )

    def updateLabel(row):
      if ("Binary Correlation" in row.keys() and row["Binary Correlation"] == "Uncorrelated"):
        row["Prediction"] = row["Binary Correlation"]
      else:
        row["Prediction"] = row["Semantic Prediction"]
      return row

    correlation_pred_data = pred.drop(pred[pred["Binary Correlation"] != "Uncorrelated"].index, axis=0)
    correlation_pred_data = correlation_pred_data.apply(updateLabel, axis=1)    
    
    data_without_uncorrelated = pred.drop(pred[pred["Binary Correlation"] == "Uncorrelated"].index, axis=0)
    semantic_input = self.loader.tokenize(data_without_uncorrelated[key_list[0]], data_without_uncorrelated[key_list[1]])

    semantic = self.model.predict(semantic_input, verbose=1)
    semantic_prediction = [self.get_label(arr, self.config["master"]) for arr in semantic]
    semantic_prediction = pd.DataFrame(semantic_prediction, columns=["Semantic Prediction"])
    semantic_pred = pd.concat([data_without_uncorrelated.reset_index(drop=True), semantic_prediction.reset_index(drop=True)], axis=1)
    semantic_pred.to_csv(os.path.join(self.config["save_path"], "semantic.csv"))

    semantic_conf_matrix = confusion_matrix(
      y_true=semantic_pred[self.config["master"]["label_key"]],
      y_pred=semantic_pred["Semantic Prediction"],
      labels=self.config["master"]["labels"].split("_")
    )
    semantic_classification_report = classification_report(
      y_true=semantic_pred[self.config["master"]["label_key"]],
      y_pred=semantic_pred["Semantic Prediction"],
      labels=self.config["master"]["labels"].split("_")
    )
    semantic_accuracy_score = accuracy_score(
      y_true=semantic_pred[self.config["master"]["label_key"]],
      y_pred=semantic_pred["Semantic Prediction"]
    )
    semantic_f1_score = f1_score(
      y_true=semantic_pred[self.config["master"]["label_key"]],
      y_pred=semantic_pred["Semantic Prediction"],
      labels=self.config["master"]["labels"].split("_"),
      average='weighted'
    )

    semantic_pred = semantic_pred.apply(updateLabel, axis=1)
    correlation_pred_data.drop(["Binary Correlation"], axis=1, inplace=True)
    semantic_pred.drop(["Semantic Prediction"], axis=1, inplace=True)

    final_pred = pd.concat([correlation_pred_data.reset_index(drop=True), semantic_pred.reset_index(drop=True)])
    final_pred.to_csv(os.path.join(self.config["save_path"], "final.csv"))

    final_conf_matrix = confusion_matrix(
      y_true=final_pred[self.config["master"]["label_key"]],
      y_pred=final_pred["Prediction"],
      labels=self.config["master"]["final_label"].split("_")
    )
    final_classification_report = classification_report(
      y_true=final_pred[self.config["master"]["label_key"]],
      y_pred=final_pred["Prediction"],
      labels=self.config["master"]["final_label"].split("_")
    )
    final_accuracy_score = accuracy_score(
      y_true=final_pred[self.config["master"]["label_key"]],
      y_pred=final_pred["Prediction"]
    )
    final_f1_score = f1_score(
      y_true=final_pred[self.config["master"]["label_key"]],
      y_pred=final_pred["Prediction"],
      labels=self.config["master"]["final_label"].split("_"),
      average='weighted'
    )

    print("====\tBinary Classification\t====")
    print("Labels: {}".format(self.config["master_binary"]["labels"].split("_")))
    print("Confusion Matrix:")
    print(relation_conf_matrix)
    print("Classification Report")
    print(relation_classification_report)
    print("F1Score :", relation_f1_score)
    print("Accuracy Score:", relation_accuracy_score)
    print("====\tSemantic Classification\t====")
    print("Labels: {}".format(self.config["master"]["labels"].split("_")))
    print("Confusion Matrix:")
    print(semantic_conf_matrix)
    print("Classification Report")
    print(semantic_classification_report)
    print("F1Score :", semantic_f1_score)
    print("Accuracy Score:", semantic_accuracy_score)
    print("====\tMerged Classification\t====")
    print("Labels: {}".format(self.config["master"]["final_label"].split("_")))
    print("Confusion Matrix:")
    print(final_conf_matrix)
    print("Classification Report")
    print(final_classification_report)
    print("F1Score :", final_f1_score)
    print("Accuracy Score:", final_accuracy_score)