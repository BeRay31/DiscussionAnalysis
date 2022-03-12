import argparse


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--run_type", required=True, help="Select one of model type shallow (SVM) | deep (IndoBERT | XLNet) based on config file"
  )
  parser.add_argument("--config", required=True, help="config path for models")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_arguments()
  if args.run_type.lower() == "shallow":
    from src.trainer.shallow import main
    main(args.config)