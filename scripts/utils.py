import os
import re
import json
import shutil
import torch
import yaml


def create_new_exp_folder(base_dir="results", prefix="exp"):
    os.makedirs(base_dir, exist_ok=True)  

    # 正则提取已有 exp 文件夹中的编号
    exp_dirs = [d for d in os.listdir(base_dir) if re.match(f"{prefix}[0-9]+$", d)]
    exp_nums = [int(re.findall(r"\d+", d)[0]) for d in exp_dirs] if exp_dirs else []

    new_exp_id = max(exp_nums) + 1 if exp_nums else 0
    new_exp_path = os.path.join(base_dir, f"{prefix}{new_exp_id}")
    os.makedirs(new_exp_path)

    print(f"新实验文件夹已创建: {new_exp_path}")
    return new_exp_path, new_exp_id


def extract_from_coco(coco_root="../coco", output_root=None, percent=0.1):
    subsets = ["train2017", "val2017", "test2017"]

    if output_root is None:
        output_root = f"../coco_{int(percent * 100)}p"

    if not os.path.exists(output_root):
        for subset in subsets:
            image_dir = os.path.join(coco_root, subset)
            ann_file = f"captions_{subset}.json"
            ann_path = os.path.join(coco_root, "annotations", ann_file)

            output_img_dir = os.path.join(output_root, subset)
            output_ann_dir = os.path.join(output_root, "annotations")
            os.makedirs(output_img_dir, exist_ok=True)
            os.makedirs(output_ann_dir, exist_ok=True)

            if not os.path.exists(ann_path):
                print(f"⚠️ Skipping {subset}({ann_file} not found)")
                # 拷贝图像即可
                image_files = sorted(os.listdir(image_dir))
                total = int(len(image_files) * percent)
                for file_name in image_files[:total]:
                    shutil.copy2(os.path.join(image_dir, file_name), os.path.join(output_img_dir, file_name))
                continue

            with open(ann_path, 'r') as f:
                data = json.load(f)

            images = sorted(data["images"], key=lambda x: x["file_name"])
            total = int(len(images) * percent)
            images = images[:total]
            image_ids = set(img["id"] for img in images)

            annotations = [ann for ann in data["annotations"] if ann["image_id"] in image_ids]

            for img in images:
                src = os.path.join(image_dir, img["file_name"])
                dst = os.path.join(output_img_dir, img["file_name"])
                shutil.copy2(src, dst)

            new_data = {
                "info": data.get("info", {}),
                "licenses": data.get("licenses", []),
                "images": images,
                "annotations": annotations
            }

            out_ann_path = os.path.join(output_ann_dir, ann_file)
            with open(out_ann_path, "w") as f:
                json.dump(new_data, f, indent=2)

            print(f"✅ {subset} processed, {len(images)} images extracted.")
    
    print(f"✅ {percent} of coco images extracted.")
    return output_root


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
    history_path = os.path.join(save_path, "history.json")
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []


def save_history(history, save_path):
    with open(os.path.join(save_path, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)