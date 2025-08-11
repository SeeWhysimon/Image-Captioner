import torch
import json

from itertools import chain
from torch.utils.data import DataLoader

from scripts import builder
from scripts.data import CocoDataset
from scripts.model import CNNEncoder, RNNDecoder
from scripts.engine import CaptionCollator, train
from scripts.vocabulary import Vocabulary
from scripts.utils import (load_configs, 
                           load_checkpoint, 
                           load_history, 
                           create_new_exp_folder, 
                           save_history)

def training_pipeline(train_config, data_config, model_config, device):
    # load training configs
    train_config = load_configs("configs/train.yaml")
        
    # build vocabulary and dataset
    torch.serialization.add_safe_globals([Vocabulary])

    with open(data_config["datasets"]["train"]["ann_path"], "r") as f:
        data = json.load(f)
    train_captions = [ann["caption"] for ann in data["annotations"]]

    if train_config["checkpoint_path"]:
        checkpoint = load_checkpoint(checkpoint_path=train_config["checkpoint_path"], 
                                     map_location=device)
        vocab = checkpoint["vocab"]
    else:
        vocab = Vocabulary()
        vocab.build(train_captions)

    collator = CaptionCollator(vocab=vocab, 
                               padding_value=data_config["dataloader"]["padding_value"])
        
    transform = builder.build_transform(data_config["transform"])
    train_set = CocoDataset(image_dir=data_config["datasets"]["train"]["image_dir"], 
                            ann_path=data_config["datasets"]["train"]["ann_path"], 
                            transform=transform)
    train_loader = DataLoader(dataset=train_set, 
                              batch_size=data_config["dataloader"]["batch_size"], 
                              shuffle=data_config["dataloader"]["shuffle"], 
                              num_workers=data_config["dataloader"]["num_workers"], 
                              collate_fn=collator, 
                              pin_memory=data_config["dataloader"]["pin_memory"])

    # build models
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

    # prepare save folder
    start_epoch = checkpoint["start_epoch"] if train_config["checkpoint_path"] else 0

    save_path = create_new_exp_folder(base_dir=train_config["save_dir"], 
                                      mode="train")
        
    # start training
    new_history = train(encoder=encoder, 
                        decoder=decoder, 
                        dataloader=train_loader, 
                        vocab=vocab, 
                        criterion=criterion, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        num_epochs=train_config["num_epochs"], 
                        print_every=train_config["print_every"], 
                        save_path=save_path, 
                        save_every=train_config["save_every"], 
                        start_epoch=start_epoch, 
                        clip_params=trainable_params, 
                        clip_max_norm=train_config["misc"]["clip_grad_norm"], 
                        device=device, 
                        history=history)
        
    save_history(history=new_history, save_path=save_path)