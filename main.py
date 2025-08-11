import os
import json
import argparse
import torch

from scripts.utils import load_configs
from scripts.pipeline import training_pipeline, testing_pipeline


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", type=str, default="train", help="Working mode.")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    model_config = load_configs("configs/model.yaml")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================ train mode ============================
    if args.mode == "train":
        train_config = load_configs("configs/train.yaml")
        training_pipeline(train_config=train_config, 
                          model_config=model_config, 
                          device=device)


    # ============================ test mode ============================
    elif args.mode == "test":
        test_config = load_configs("configs/test.yaml")
        
        testing_pipeline(test_config=test_config, 
                         model_config=model_config, 
                         device=device)

    else:
        print(f"Unsupported mode: {args.mode}.")