import preprocessor as tp
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm
import string
import re
import os

class DataPreprocessor:
  def __init__(self, dataFrame, save_path, stem = False):
    self.dataFrame = dataFrame
    self.stem = stem
    if stem:
      self.stemmer = StemmerFactory().create_stemmer()
    self.save_path = save_path
    tp.set_options(tp.OPT.URL, tp.OPT.MENTION)


  def preprocess(self, text):
    # Clean Tweet
    tempText = tp.clean(text)
    if self.stem:
      # Stem text, clean emoji, punctuation, and lowercase, clean trailing spaces
      tempText = self.stemmer.stem(tempText)
    else:
      # Handle Hashtag
      tempText = tempText.replace("#","").replace("\n", " ")
      # Remove Punctuation
      tempText = tempText.translate(str.maketrans('', '', string.punctuation))
      # Lowercase text
      tempText = tempText.lower()
      # Clean multiple spaces
      tempText = re.sub('\\s+', ' ', tempText)
    return tempText

  def main(self):
    for idx, row in tqdm(self.dataFrame.iterrows(), total=self.dataFrame.shape[0]):
      self.dataFrame.loc[idx, "Tweet"] = self.preprocess(self.dataFrame.loc[idx, "Tweet"])
      self.dataFrame.loc[idx, "Comment"] = self.preprocess(self.dataFrame.loc[idx, "Comment"])
    self.dataFrame.dropna(inplace=True)
    self.dataFrame.to_csv(os.path.join(self.save_path, f"{os.path.basename(self.save_path)}_preprocessed.csv"))
    return self.dataFrame