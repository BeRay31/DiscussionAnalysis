import preprocessor as tp
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from regex import I
from tqdm import tqdm
import string
import re
import os

class DataPreprocessor:
  def __init__(self, dataFrame, save_path, save_name="train", stem = False, handle_sarcasm = "Normal"):
    self.dataFrame = dataFrame
    self.stem = stem
    if stem:
      self.stemmer = StemmerFactory().create_stemmer()
    self.save_path = save_path
    self.save_name = save_name
    self.handle_sarcasm = handle_sarcasm
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
    # Handle Sarcasm Label
    i_pro_sarcasm = self.dataFrame[self.dataFrame["Label"] == "Pro Sarcasm"].index
    i_cont_sarcasm = self.dataFrame[self.dataFrame["Label"] == "Contra Sarcasm"].index
    if self.handle_sarcasm == "Delete":
      self.dataFrame.drop(i_pro_sarcasm, inplace=True)
      self.dataFrame.drop(i_cont_sarcasm, inplace=True)
    elif self.handle_sarcasm == "Replace":
      self.dataFrame.loc[i_pro_sarcasm, "Label"] = "Pro"
      self.dataFrame.loc[i_cont_sarcasm, "Label"] = "Contra"
    self.dataFrame.dropna(inplace=True)
    self.dataFrame.to_csv(os.path.join(self.save_path, f"{self.save_name}_preprocessed.csv"))
    return self.dataFrame