import tensorflow as tf
from transformers import TFBertModel, XLNetModel

tf.random.set_seed(13518136)
from tensorflow.keras import Model

class DeepClassifier(Model):
  def __init__(self, config):
    super().__init__()
    self.config = config

    if self.config["type"] == "bert":
      self.embedder = TFBertModel.from_pretrained(config["model_name"])
    elif config["type"].lower() == "xlnet":
      self.embedder = XLNetModel.from_pretrained(config["model_name"])