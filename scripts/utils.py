import os
import time
import json
import torch
import yaml


def create_new_exp_folder(base_dir="logs", mode=None):
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    new_exp_path = os.path.join(base_dir, mode, timestamp)
    os.makedirs(new_exp_path)
    
    print(f"Experiment folder created: {new_exp_path}.")
    return new_exp_path
    

def load_checkpoint(checkpoint_path, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location)
    
    start_epoch = checkpoint["epoch"] + 1
    encoder_state_dict = checkpoint["encoder"]
    decoder_state_dict = checkpoint["decoder"]
    optimizer_state_dict = checkpoint.get("optimizer", None)
    scheduler_state_dict = checkpoint.get("scheduler", None)
    vocab = checkpoint.get("vocab", None)

    return {
        "start_epoch": start_epoch,
        "encoder_state_dict": encoder_state_dict,
        "decoder_state_dict": decoder_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "vocab": vocab
    }


def load_configs(*files):
    config = {}
    for path in files:
        with open(path, 'r') as f:
            config.update(yaml.safe_load(f))
    return config


def load_history(history_path):
    if history_path:
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []


def save_history(history, save_path):
    with open(os.path.join(save_path, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)