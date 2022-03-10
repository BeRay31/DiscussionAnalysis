import argparse


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--type", required=True, help="Select one of model type [SVM | IndoBERT | XLNet]"
  )
  parser.add_argument(
    "--embedding", required=True, help="Select one of embedding type [Word2Vec | FastText]"
  )
  parser.add_argument("--prefix", required=True, help="prefix name for trained models")
  parser.add_argument("--config", required=True, help="config path for models")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_arguments()
  print(args.type)
  print(args.prefix)
  print(args.config)
  # from src.trainer.pretrained import main
  # main(args.config, args.prefix)