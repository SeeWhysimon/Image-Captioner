import torch
import torch.nn as nn
import os

from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from utils import generate_captions

class CaptionCollator:
    def __init__(self, vocab, padding_value):
        self.vocab = vocab
        self.padding_value = padding_value

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        targets = [self.vocab.encode(c) for c in captions]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.padding_value)
        return images, targets

def train(encoder, 
          decoder, 
          dataloader, 
          vocab, 
          optimizer, 
          criterion, 
          scheduler=None,
          num_epochs=5, 
          print_every=1, 
          save_path=None, 
          save_every=10, 
          start_epoch=0, 
          device='cpu'):
    encoder.train()
    decoder.train()
    history = [] 
    
    print(f"Training on {device}...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0
        for imgs, caps in dataloader:
            imgs, caps = imgs.to(device), caps.to(device)

            features = encoder(imgs)
            outputs = decoder(features, caps)

            loss = criterion(outputs.reshape(-1, len(vocab)), caps[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(decoder.parameters()) + 
                                     list(encoder.backbone[-1].parameters()) + 
                                     list(encoder.classifier.parameters()), 
                                     max_norm=5)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        
        if scheduler:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        
        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": current_lr
        })

        if (epoch + 1) % print_every == 0:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.6f} | LR: {current_lr:.6e}")

        if save_path and (epoch + 1) % save_every == 0:
            model_name = os.path.join(save_path, f"caption_model_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(), 
                "scheduler": scheduler.state_dict() if scheduler else None,
                "vocab": vocab
            }, model_name)
            print(f"已保存模型到 {model_name}")
            
    return history

def evaluate_bleu(encoder, decoder, dataloader, vocab, device="cpu"):
    encoder.eval()
    decoder.eval()

    smoothie = SmoothingFunction().method4
    total_scores = {"BLEU-1": [], "BLEU-2": [], "BLEU-3": [], "BLEU-4": []}

    with torch.no_grad():
        for images, captions_gt in dataloader:
            images = images.to(device)

            # === Step 1: 生成预测句子（List[str]）
            predicted_captions = generate_captions(encoder, decoder, images, vocab, device=device)

            # === Step 2: 对每个样本计算 BLEU
            for pred_sentence, gt_tokens in zip(predicted_captions, captions_gt):
                candidate = pred_sentence.split()  
                reference = [vocab.decode(gt_tokens).split()]  

                for i in range(1, 5):
                    weight = tuple([1/i] * i)  
                    score = sentence_bleu(reference, candidate, weights=weight, smoothing_function=smoothie)
                    total_scores[f"BLEU-{i}"].append(score)

    final_scores = {k: sum(v) / len(v) if v else 0.0 for k, v in total_scores.items()}
    return final_scores