import preprocessor as tp
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm
import pandas as pd
import re
import os

class DataPreprocessor:
  def __init__(self, dataFrame, save_path, stem = False):
    self.dataFrame = dataFrame
    self.stem = stem
    self.save_path = save_path
    tp.set_options(tp.OPT.URL, tp.OPT.MENTION, tp.OPT.URL, tp.OPT.EMOJI, tp.OPT.SMILEY)


  def preprocess(self, text):
    tempText = tp.tokenize(text)
    tempText = re.sub(r'[^\w\s]', '', tempText)
    if self.stem:
      stemmer = StemmerFactory().create_stemmer()
      tempText = stemmer.stem(tempText)
    return tempText

  def main(self):
    for idx, row in tqdm(self.dataFrame.iterrows(), total=self.dataFrame.shape[0]):
      self.dataFrame.loc[idx, "Tweet"] = self.preprocess(self.dataFrame.loc[idx, "Tweet"])
      self.dataFrame.loc[idx, "Comment"] = self.preprocess(self.dataFrame.loc[idx, "Comment"])
    self.dataFrame.dropna(inplace=True)
    self.dataFrame.to_csv(os.path.join(self.save_path, f"{os.path.basename(self.save_path)}_preprocessed.csv"))
    return self.dataFrame