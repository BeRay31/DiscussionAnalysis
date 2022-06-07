import numpy as np
from src.classifier import DeepClassifier
from src.loader import DeepLoader
import tensorflow as tf

class DeepPredictor:
  def __init__(self, model_path):
    self.labels = "Uncorrelated_Pro_Neutral_Contra"

    self.model = DeepClassifier(
      { 
       "type": "roberta",
       "model_name": "cardiffnlp/twitter-xlm-roberta-base",
       "recurrent_layer": "bilstm",
       "recurrent_unit": 64,
       "recurrent_dropout": 0.4,
       "labels": self.labels
      }
    )
    self.model_path = model_path
    self.loader = DeepLoader(
      {
        "tokenizer_type": "roberta",
        "model_name": "cardiffnlp/twitter-xlm-roberta-base",
        "labels": self.labels,
        "label_key": "Label",
        "key_list": "Tweet_Comment",
        "data_path": False,
        "tokenizer_config": {
          "max_length": 512,
          "padding": "max_length",
          "return_tensors": "tf",
          "truncation": True
        }
      }
    )
  
  def get_label(self, arr):
    labels = self.labels.split("_")
    idx = np.argmax(arr)
    return labels[idx]
  
  def predict_single(self, tweet, retweet):
    # prepare data
    data = self.loader.tokenize_single(tweet, retweet)
    self.model.build([data])
    self.model.load_weights(self.model_path)
    pred = self.model.predict(
      [data],
      batch_size=4,
      verbose=1,
    )
    return self.get_label(pred)

  def predict_batch(self, tweet, retweets):
    data = self.loader.tokenize_batch(tweet, retweets)
    pred = self.model.predict(
      data,
      batch_size=4,
      verbose=1,
    )
    return [self.get_label(arr) for arr in pred]
  