import torch
import json
import os

from itertools import chain
from torch.utils.data import DataLoader
from glob import glob

from scripts import builder
from scripts.data import CocoDataset, TestDataset
from scripts.model import CNNEncoder, RNNDecoder
from scripts.engine import CaptionCollator, train, inference, test_collate_fn
from scripts.vocabulary import Vocabulary
from scripts.utils import (load_configs, 
                           load_checkpoint, 
                           load_history, 
                           create_new_exp_folder, 
                           save_history)


def training_pipeline(train_config, model_config, device):
    # loading training configs
    train_config = load_configs("configs/train.yaml")
        
    # building vocabulary
    torch.serialization.add_safe_globals([Vocabulary])

    with open(train_config["data"]["ann_path"], "r") as f:
        data = json.load(f)
    train_captions = [ann["caption"] for ann in data["annotations"]]

    if train_config["checkpoint_path"]:
        checkpoint = load_checkpoint(checkpoint_path=train_config["checkpoint_path"], 
                                     map_location=device)
        vocab = checkpoint["vocab"]
    else:
        vocab = Vocabulary()
        vocab.build(train_captions)

    #building dataset
    collator = CaptionCollator(vocab=vocab, 
                               padding_value=train_config["dataloader"]["padding_value"])
        
    transform = builder.build_transform(train_config["transform"])
    train_set = CocoDataset(image_dir=train_config["data"]["image_dir"], 
                            ann_path=train_config["data"]["ann_path"], 
                            transform=transform)
    train_loader = DataLoader(dataset=train_set, 
                              batch_size=train_config["dataloader"]["batch_size"], 
                              shuffle=train_config["dataloader"]["shuffle"], 
                              num_workers=train_config["dataloader"]["num_workers"], 
                              collate_fn=collator, 
                              pin_memory=train_config["dataloader"]["pin_memory"])

    # building models
    encoder = CNNEncoder(embed_size=model_config["encoder"]["embed_size"], 
                         backbone=model_config["encoder"]["backbone"]["type"]).to(device)
    decoder = RNNDecoder(embed_size=model_config["decoder"]["embed_size"], 
                         hidden_size=model_config["decoder"]["hidden_size"], 
                         vocab_size=len(vocab), 
                         num_layers=model_config["decoder"]["num_layers"]).to(device)
        
    if train_config["checkpoint_path"]:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])

    trainable_params = list(filter(lambda p: p.requires_grad, 
                            chain(encoder.parameters(), decoder.parameters())))
        
    criterion = builder.build_criterion(cfg=train_config)
    optimizer = builder.build_optimizer(cfg=train_config, 
                                        model_or_params=trainable_params)
    scheduler = builder.build_scheduler(cfg=train_config, 
                                        optimizer=optimizer)
        
    if train_config["checkpoint_path"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    history = load_history(train_config["history_path"])

    # preparing save folder
    start_epoch = checkpoint["start_epoch"] if train_config["checkpoint_path"] else 0

    exp_dir = create_new_exp_folder(base_dir=train_config["save_dir"], 
                                      mode="train")
        
    # starting training
    new_history = train(encoder=encoder, 
                        decoder=decoder, 
                        dataloader=train_loader, 
                        vocab=vocab, 
                        criterion=criterion, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        num_epochs=train_config["num_epochs"], 
                        print_every=train_config["print_every"], 
                        save_path=exp_dir, 
                        save_every=train_config["save_every"], 
                        start_epoch=start_epoch, 
                        clip_params=trainable_params, 
                        clip_max_norm=train_config["misc"]["clip_grad_norm"], 
                        device=device, 
                        history=history)
        
    save_history(history=new_history, save_path=exp_dir)

    print(f"[INFO] Training finished. Logs and models are saved to {exp_dir}")


def testing_pipeline(test_config, model_config, device):
    if test_config["checkpoint_path"] is None:
        raise ValueError("Checkpoint path must be provided for testing.")
    
    if test_config["data"]["image_dir"] is None:
        raise ValueError("Image directory must be provided for testing.")
    
    # loading checkpoint
    torch.serialization.add_safe_globals([Vocabulary])
    checkpoint = load_checkpoint(checkpoint_path=test_config["checkpoint_path"],
                                 map_location=device)
    
    # building vocabulary
    vocab = checkpoint["vocab"]

    # building test dataloader
    test_image_dir = test_config["data"]["image_dir"]
    test_image_paths = glob(os.path.join(test_image_dir, "*.jpg")) + \
                       glob(os.path.join(test_image_dir, "*.png"))
    transform = builder.build_transform(test_config["transform"])
    test_set = TestDataset(image_paths=test_image_paths, 
                           transform=transform)
    test_loader = DataLoader(dataset=test_set, 
                              batch_size=test_config["dataloader"]["batch_size"], 
                              shuffle=test_config["dataloader"]["shuffle"], 
                              num_workers=test_config["dataloader"]["num_workers"], 
                              collate_fn=test_collate_fn, 
                              pin_memory=test_config["dataloader"]["pin_memory"])
    
    # building models
    encoder = CNNEncoder(embed_size=model_config["encoder"]["embed_size"],
                         backbone=model_config["encoder"]["backbone"]["type"]).to(device)
    decoder = RNNDecoder(vocab_size=len(vocab),
                         embed_size=model_config["decoder"]["embed_size"],
                         hidden_size=model_config["decoder"]["hidden_size"],
                         num_layers=model_config["decoder"]["num_layers"]).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    decoder.eval()

    # inference
    results = inference(encoder, decoder, test_loader, vocab, max_len=20, device=device)

    # storing results
    exp_dir = create_new_exp_folder(test_config["save_dir"], mode="test")
    checkpoint_name = test_config["checkpoint_path"].split("/")[-1].split(".")[0]
    caption_save_path = os.path.join(exp_dir, f"{checkpoint_name}_captions.json")

    with open(caption_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Test finished. Results saved to {exp_dir}")