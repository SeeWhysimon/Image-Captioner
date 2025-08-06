import torch
import torch.nn as nn
import os

import config

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from PIL import Image

from scripts.data import CocoDataset
from scripts.vocabulary import Vocabulary
from scripts.utils import (build_transform,
                           create_new_exp_folder,
                           extract_from_coco,
                           generate_captions,
                           load_checkpoint,  
                           load_configs,
                           load_history, 
                           save_history)
from scripts.engine import CaptionCollator, train, evaluate_bleu
from scripts.model import CNNEncoder, RNNDecoder
from scripts.builder import build_optimizer, build_scheduler

if __name__ == "__main__":
    print(f"🚀 Working on {config.device}...")

    data_cfg = load_configs("configs/data.yaml")
    model_cfg = load_configs("configs/model.yaml")
    train_cfg = load_configs("configs/train.yaml")

    if config.mode == "train":
        transform = build_transform(data_cfg["transform"])
        dataset = CocoDataset(image_dir=data_cfg["datasets"]["train"]["image_dir"], 
                              ann_path=data_cfg["datasets"]["train"]["ann_path"], 
                              transform=transform)
        captions_all = [c for _, c in dataset.samples]

        if train_cfg["checkpoint_path"] and os.path.exists(train_cfg["checkpoint_path"]):
            print(f"🔁 Loading checkpoint: {train_cfg['checkpoint_path']}")
            ckpt = load_checkpoint(train_cfg["checkpoint_path"], 
                                   map_location=train_cfg["device"])

            vocab = ckpt["vocab"]
            if vocab is None:
                raise ValueError("❌ Checkpoint 中缺失 vocab, 无法恢复训练")

            start_epoch = ckpt["start_epoch"]
            exp_path = os.path.dirname(config.checkpoint_path)
            print(f"📂 Inheritting folder: {exp_path}")
            history = load_history(exp_path)
        else:
            vocab = Vocabulary()
            vocab.build(captions_all)
            start_epoch = 0
            history = []
            exp_path, _ = create_new_exp_folder(base_dir=config.save_path)

        collate = CaptionCollator(vocab, padding_value=data_cfg["dataloader"]["padding_value"])
        dataloader = DataLoader(
            dataset,
            batch_size=data_cfg["dataloader"]["batch_size"],
            shuffle=data_cfg["dataloader"]["shuffle"],
            collate_fn=collate,
            num_workers=data_cfg["dataloader"]["num_workers"],
            pin_memory=data_cfg["dataloader"]["pin_memory"]
        )

        encoder = CNNEncoder(model_cfg["encoder"]["embed_size"], 
                             model_cfg["encoder"]["backbone"]).to(train_cfg["device"])
        decoder = RNNDecoder(model_cfg["decoder"]["embed_size"], 
                             model_cfg["decoder"]["hidden_size"], 
                             len(vocab)).to(train_cfg["device"])

        if config.checkpoint_path and os.path.exists(config.checkpoint_path):
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            decoder.load_state_dict(ckpt["decoder_state_dict"])

        param_list = (list(decoder.parameters()) +
                      list(encoder.backbone[-1].parameters()) +
                      list(encoder.classifier.parameters()))

        criterion = nn.CrossEntropyLoss(ignore_index=config.padding_idx)
        optimizer = build_optimizer(train_cfg, param_list)
        scheduler = build_scheduler(train_cfg, optimizer)

        if config.checkpoint_path and os.path.exists(config.checkpoint_path):
            if ckpt["optimizer_state_dict"]:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt["scheduler_state_dict"]:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        new_history = train(
            encoder,
            decoder,
            dataloader,
            vocab,
            optimizer,
            criterion,
            scheduler,
            config.num_epochs,
            config.print_every,
            exp_path,
            config.save_every,
            start_epoch, 
            config.device
        )

        history.extend(new_history)
        save_history(history, exp_path)
        print(f"✅ 模型已保存到：{exp_path}")
    
    elif config.mode == "evaluate":
        torch.serialization.add_safe_globals([Vocabulary])

        ckpt = load_checkpoint(config.checkpoint_path, map_location=config.device)
        vocab = ckpt["vocab"]
        if vocab is None:
            raise ValueError("❌ Checkpoint 中未保存 vocab, 无法继续评估")

        encoder = CNNEncoder(config.embed_size, config.backbone).to(config.device)
        decoder = RNNDecoder(config.embed_size, config.hidden_size, len(vocab)).to(config.device)

        encoder.load_state_dict(ckpt["encoder_state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])

        data_root = extract_from_coco(percent=config.percent)
        image_dir = os.path.join(data_root, "val2017")
        ann_path = os.path.join(data_root, "annotations", "captions_val2017.json")

        dataset = CocoDataset(image_dir=image_dir, ann_path=ann_path, transform=config.transform)
        collate = CaptionCollator(vocab, padding_value=config.padding_idx)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=8,
            pin_memory=True
        )

        score = evaluate_bleu(encoder, decoder, dataloader, vocab, config.device)
        print(f"BLEU-4 score: {score['BLEU-4']:.6f}")

    elif config.mode == "refer":
        ckpt = load_checkpoint(config.checkpoint_path, map_location=config.device)
        
        vocab = ckpt["vocab"]
        if vocab is None:
            raise ValueError("❌ Checkpoint 中未保存 vocab")

        encoder = CNNEncoder(config.embed_size, backbone=config.backbone).to(config.device)
        decoder = RNNDecoder(config.embed_size, config.hidden_size, len(vocab)).to(config.device)

        encoder.load_state_dict(ckpt["encoder_state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])

        encoder.eval()
        decoder.eval()

        image = Image.open(config.images).convert("RGB")
        image = config.transform(image).unsqueeze(0).to(config.device)

        result = generate_captions(encoder, decoder, image, vocab, config.max_len, config.device)
        print("Captions:")
        print(result)

    else:
        print("❌ Mode unsupported.")