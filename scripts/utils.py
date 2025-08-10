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


def generate_captions(encoder, decoder, images, vocab, max_len=20, device='cpu'):
    batch_size = images.size(0)
    with torch.no_grad():
        features = encoder(images)  # [batch_size, embed_size]
        inputs = features.unsqueeze(1)  # [batch_size, 1, embed_size]
        hidden = None
        
        # 初始化 captions（每个是 List[str]）
        captions = [['<SOS>'] for _ in range(batch_size)]
        completed = [False] * batch_size  # 是否已生成 <EOS>
        
        for _ in range(max_len):
            outputs, hidden = decoder.rnn(inputs, hidden)  # [batch_size, 1, hidden_size]
            logits = decoder.fc(outputs.squeeze(1))        # [batch_size, vocab_size]
            predicted = logits.argmax(dim=-1)              # [batch_size]
            
            for i in range(batch_size):
                if not completed[i]:
                    word = vocab.idx2word[predicted[i].item()]
                    if word == '<EOS>':
                        completed[i] = True
                    else:
                        captions[i].append(word)
            
            if all(completed):
                break
            
            inputs = decoder.embedding(predicted).unsqueeze(1)  # [batch_size, 1, embed_size]
        
        # 拼接输出，去掉 <SOS>
        final_captions = [' '.join(words[1:]) for words in captions]
        return final_captions
    

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


def load_history(save_path):
    if save_path:
        history_path = os.path.join(save_path, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
    else:
        return []


def save_history(history, save_path):
    with open(os.path.join(save_path, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)