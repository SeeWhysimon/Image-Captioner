import torch
import os
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from scripts.utils import generate_captions

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


def train_step(encoder,
               decoder,
               batch, # (imgs, caps)
               vocab,
               optimizer,
               criterion,
               device='cpu',
               clip_params=None, 
               clip_max_norm=5.0): 
    imgs, caps = batch
    imgs, caps = imgs.to(device), caps.to(device)

    # forward
    features = encoder(imgs)
    outputs = decoder(features, caps)

    loss = criterion(outputs.reshape(-1, len(vocab)),
                     caps[:, 1:].reshape(-1))

    # backward
    optimizer.zero_grad()
    loss.backward()

    if clip_params:
        nn.utils.clip_grad_norm_(parameters=clip_params, max_norm=clip_max_norm)

    optimizer.step()
    return loss.item()


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
          clip_params=None, 
          clip_max_norm=5.0,
          device='cuda', 
          history=[]):
    encoder.to(device)
    decoder.to(device)
    if hasattr(criterion, 'to'):
        criterion.to(device)

    encoder.train()
    decoder.train()
    
    print(f"Training on {device}...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0.0

        for batch in dataloader:
            batch_loss = train_step(encoder=encoder, 
                                    decoder=decoder, 
                                    batch=batch, 
                                    vocab=vocab, 
                                    optimizer=optimizer, 
                                    criterion=criterion, 
                                    device=device, 
                                    clip_params=clip_params, 
                                    clip_max_norm=clip_max_norm)
            total_loss += batch_loss

        avg_loss = total_loss / len(dataloader)

        if scheduler:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": current_lr
        })

        if (epoch + 1) % print_every == 0 or (epoch + 1) == (start_epoch + num_epochs):
            print(f"[Epoch {epoch+1} / {start_epoch+num_epochs}] Loss: {avg_loss:.6f} | LR: {current_lr:.6e}")

        if save_path and ((epoch + 1) % save_every == 0 or (epoch + 1) == (start_epoch + num_epochs)):
            model_name = os.path.join(save_path, f"caption_model_{epoch+1}epochs.pth")
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "vocab": vocab
            }, model_name)
            print(f"Model saved as {model_name}.")

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