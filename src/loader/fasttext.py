import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
import time
import numpy as np
import pandas as pd
import gensim


# Class For OOV
class FastText:
  def __init__(self, vector_length = 100, max_sequence_length = 101, model_path = None, oov_log = False, verbose = False):
    self.model = None
    self.vector_length = vector_length
    self.max_sequence_length = max_sequence_length
    self.padding = np.zeros(vector_length,)
    self.with_verbose = verbose
    self.enable_oov_log = oov_log
    if oov_log:
      self.oov_log = []
    if model_path:
      self.model = gensim.models.Word2Vec.load(model_path)
  
  def text_to_vector(self, text, disable_log = False):
    if not bool(self.model):
      raise ValueError("There is no model loaded")
    if self.with_verbose and not disable_log:
      start = time.time()
    # Tokenize
    tokens = word_tokenize(text)
    # Normalize the final vector based on sentence length
    vector_length = len(tokens) if len(tokens) < self.max_sequence_length else self.max_sequence_length
    vector = []
    for token in tokens[:vector_length]:
      try:
        word_vector = self.model.wv.get_vector(token)
      except KeyError:
        word_vector = self.padding
        if self.enable_oov_log:
          self.oov_log.append(token)
      vector.append(word_vector)
    
    pad = self.max_sequence_length - len(vector)
    for i in range(pad):
      vector.append(self.padding)
    if self.with_verbose and not disable_log:
      end = time.time()
    
    final_vector = np.asarray(vector).flatten() 
    if (self.with_verbose and not disable_log):
      print("Transformation took: {} s, with vector length: {}".format(round(end-start,2), len(final_vector)))
    return final_vector
  
  def df_to_vector(self, df, keyList = [], concat = True, to_df = False):
    if self.with_verbose:
      start = time.time()
    vector = []
    for idx, row in df.iterrows():
      vector_temp = []
      for key in keyList:
        temp = self.text_to_vector(row[key], self.with_verbose)
        if concat:
          temp = temp.tolist()
        if len(vector_temp) == 0:
          vector_temp = temp
        else:
          vector_temp += temp
      vector.append(vector_temp)
    
    vector = np.asarray(vector)
    if self.with_verbose:
      end = time.time()
    res = vector
    if to_df:
      res = pd.DataFrame(res, columns=["Vectors"])
    if self.with_verbose:
      print("Transformation dataframe took {}, with total vectors {}".format(round(end-start,2), len(res)))
    return res

  # Assumed already preprocessed
  def fit(self, df, key_list = []):
    corpus = []
    for key in key_list:
      corpus += df[key].unique().tolist()
    tokenized_corp = [word_tokenize(sent) for sent in corpus]
    if not self.model:
      self.model = gensim.models.Word2Vec(tokenized_corp, vector_size=self.vector_length, window=5, min_count=1, workers=4, seed=13518136)
    else:
      self.model.build_vocab(tokenized_corp)
      self.model.train(tokenized_corp, total_examples=self.model.corpus_count, epochs=self.model.epochs)


