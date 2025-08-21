import torch
import os
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class CaptionCollator:
    def __init__(self, vocab, padding_value):
        self.vocab = vocab
        self.padding_value = padding_value

    def __call__(self, batch):
        images, captions, image_paths = zip(*batch)
        images = torch.stack(images)
        targets = [self.vocab.encode(c) for c in captions]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.padding_value)
        return images, targets, image_paths
    

def test_collate_fn(batch):
    # batch: list of (image_tensor, image_path_str)
    images, paths = zip(*batch)          # tuple of tensors, tuple of strs
    images = torch.stack(images, dim=0)  # [B, C, H, W]
    return images, list(paths)           


def train_step(encoder,
               decoder,
               batch, # (imgs, caps, img_paths)
               vocab,
               optimizer,
               criterion,
               device='cpu',
               clip_params=None, 
               clip_max_norm=5.0): 
    imgs, caps, _ = batch
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
          criterion, 
          optimizer, 
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
    encoder.train()
    decoder.train()
    
    print(f"[INFO] Training on {device}...")
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
            print(f"[INFO] Model saved as {model_name}.")

    return history


def inference(encoder, decoder, dataloader, vocab, max_len=20, device='cpu'):
    encoder.eval(); decoder.eval()
    results = []
    with torch.no_grad():
        for images, paths in dataloader:  # paths: list[str]
            images = images.to(device)
            B = images.size(0)

            features = encoder(images)             # [B, E]
            inputs = features.unsqueeze(1)         # [B, 1, E]
            hidden = None

            captions = [['<SOS>'] for _ in range(B)]
            completed = [False] * B

            for _ in range(max_len):
                outputs, hidden = decoder.rnn(inputs, hidden)   # [B, 1, H]
                logits = decoder.fc(outputs.squeeze(1))         # [B, V]
                predicted = logits.argmax(dim=-1)               # [B]

                for i in range(B):
                    if not completed[i]:
                        w = vocab.idx2word[predicted[i].item()]
                        if w == '<EOS>':
                            completed[i] = True
                        else:
                            captions[i].append(w)

                if all(completed): break
                inputs = decoder.embedding(predicted).unsqueeze(1)  # [B, 1, E]

            # dropping <SOS>
            finals = [' '.join(ws[1:]) for ws in captions]  

            for i, cap in enumerate(finals):
                results.append({"image": paths[i], "caption": cap})  
    return results


def evaluate_bleu(encoder, decoder, dataloader, vocab, max_len=20, device="cpu"):
    encoder.eval()
    decoder.eval()

    smoothie = SmoothingFunction().method4
    results = []

    with torch.no_grad():
        for images, captions_gt, image_paths in dataloader:  
            images = images.to(device)
            B = images.size(0)

            features = encoder(images)              # [B, E]
            inputs = features.unsqueeze(1)          # [B, 1, E]
            hidden = None

            captions = [['<SOS>'] for _ in range(B)]
            completed = [False] * B

            for _ in range(max_len):
                outputs, hidden = decoder.rnn(inputs, hidden)   # [B, 1, H]
                logits = decoder.fc(outputs.squeeze(1))         # [B, V]
                predicted = logits.argmax(dim=-1)               # [B]

                for i in range(B):
                    if not completed[i]:
                        w = vocab.idx2word[predicted[i].item()]
                        if w == '<EOS>':
                            completed[i] = True
                        else:
                            captions[i].append(w)

                if all(completed):
                    break

                inputs = decoder.embedding(predicted).unsqueeze(1)  # [B, 1, E]

            pred_texts = [' '.join(ws[1:]) for ws in captions]

            for b in range(B):
                reference_text = ' '.join(vocab.decode(captions_gt[b]).split()[1:-1])  
                candidate_text = pred_texts[b]

                candidate = candidate_text.split()
                reference = [reference_text.split()]

                per_n_scores = []
                scores = {}
                scores["image"] = image_paths[b]    
                scores["reference"] = reference_text
                scores["prediction"] = candidate_text
                
                for n in range(1, 5):
                    weights = tuple([1.0 / n] * n)
                    sc = sentence_bleu(reference, candidate, weights=weights, smoothing_function=smoothie)
                    scores[f"BLEU-{n}"] = float(sc)
                    per_n_scores.append(sc)

                scores["total"] = float(sum(per_n_scores) / 4.0)

                results.append(scores)

    return results