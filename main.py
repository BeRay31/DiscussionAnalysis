import argparse
import tensorflow as tf

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--run_type", required=True, help="Select one of model type shallow (SVM) | deep (IndoBERT | XLNet) based on config file"
  )
  parser.add_argument("--config", required=True, help="config path for models")
  parser.add_argument("--disabled_gpu", required=False, help="GPU to hide using set visible devices using this template gpu1 | gpu2", default="")
  parser.add_argument("--gpu_used", required=False, help="GPU used 3 4 5")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_arguments()

  if args.disabled_gpu != "":
    disabled_gpu = []
    visible_gpu =[]
    disabled_gpu = set([int(i) for i in args.disabled_gpu.split("|")])
    physical_devices = tf.config.list_physical_devices('GPU')
    for i in range(len(physical_devices)):
      if not i in disabled_gpu:
        visible_gpu.append(physical_devices[i])
    tf.config.set_visible_devices(visible_gpu, 'GPU')
    print(f"GPU {disabled_gpu} are not visible in this train")

  if args.run_type.lower() == "shallow":
    from src.trainer.shallow import main
    main(args.config)
  elif args.run_type.lower() == "deep":
    from src.trainer.deep import main
    main(args.config, args.gpu_used)