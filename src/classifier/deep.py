import tensorflow as tf
from transformers import TFBertModel, XLNetModel

tf.random.set_seed(13518136)
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    GRU,
    Dropout,
    Bidirectional,
)
class DeepClassifier(Model):
  def __init__(self, config):
    super().__init__()
    self.config = config

    # Define Embedding Layer
    if self.config["type"] == "bert":
      self.embedder = TFBertModel.from_pretrained(config["model_name"])
    elif config["type"].lower() == "xlnet":
      self.embedder = XLNetModel.from_pretrained(config["model_name"])
    
    # Define Recurrent Layer
    if self.config["recurrent_layer"].lower() == "lstm":
      self.recurrent = LSTM(self.config["recurrent_config"])
    elif self.config["recurrent_layer"].lower() == "bilstm":
      self.recurrent = Bidirectional(LSTM(self.config["recurrent_config"]))
    elif self.config["recurrent_layer"].lower() == "gru":
      self.recurrent = GRU(self.config["recurrent_config"])
    elif self.config["recurrent_layer"].lower() == "bigru":
      self.recurrent = Bidirectional(GRU(self.config["recurrent_config"]))
    elif self.config["recurrent_layer"].lower() == "dense":
      self.recurrent = Dense(self.config["recurrent_unit"], activation="relu")
    else:
      self.recurrent = None
      raise ValueError("only support (lstm | bilstm | gru | bigru) layer type")
    
    # Define features dropout
    self.dropout = Dropout(self.config["recurrent_dropout"])
    labels = self.config["labels"].split("_")
    
    # Define Output Layer
    self.output = Dense(len(labels), activation="softmax", name="classifier")
  
  def call(self, X, training=None):
    embedding_part_taken = (
      "last_hidden_state" if self.config["recurrent_layer"].lower() in ["lstm", "bilstm", "gru", "bigru"]
      else "pooler_output"
    )

    # Pass to BERT or XLNet Model
    X_embed = self.embedder(X)[embedding_part_taken]
    
    # Passed to recurrent layer
    X_recurrent = X_embed
    if self.recurrent != None:
      X_recurrent = self.recurrent(X_embed)
    
    # Dropout features to decrease overfit
    X_ = self.dropout(X_recurrent, training=training)
    
    # Final Dense layer
    res = self.output(X_)

    return res
    

