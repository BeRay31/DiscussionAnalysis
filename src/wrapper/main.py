from src.util import get_config
from src.loader import DeepLoader
from src.classifier import DeepClassifier
import numpy as np

default_config = {
  'master': {
    'model_name': 'cardiffnlp/twitter-xlm-roberta-base',
    'labels': 'Pro_Neutral_Contra',
    'model_weight_path': 'D:\\WorkBench\\TA NLP\\res_deep\\res_53\\3_label\\final_model_3_label.h5'
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
  
  def get_label(self, arr):
    labels = self.config["master"]["labels"].split("_")
    idx = np.argmax(arr)
    return labels[idx]

  def predict(self, tweet, retweet):
    input = self.loader.tokenize(tweet, retweet)
    pred = self.model.predict(dict(input))
    return [self.get_label(item) for item in pred]

  def predict_batch_retweet(self, tweet, retweets):
    tweets = [tweet for i in range(len(retweets))]
    input = self.loader.tokenize(tweets, retweets)
    pred = self.model.predict(dict(input))
    return [self.get_label(item) for item in pred]