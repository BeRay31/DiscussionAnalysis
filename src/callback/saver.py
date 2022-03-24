from tensorflow.keras.callbacks import Callback
import os

class CustomSaver(Callback):
  def __init__(self, save_path):
    super().__init__()
    self.save_path = save_path

  def on_epoch_end(self, epoch, logs = {}):
    self.model.save_weights(os.path.join(self.save_path, "model_{}.h5".format(epoch)))