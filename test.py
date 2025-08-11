import torch
from scripts.vocabulary import Vocabulary

torch.serialization.add_safe_globals([Vocabulary])

ckpt = torch.load("D:/Image_Captioning_COCO/logs/train/20250809-205742/caption_model_10epochs.pth", map_location="cpu")
print(type(ckpt))
if isinstance(ckpt, dict):
    print("keys:", ckpt.keys())
else:
    print("不是 dict，类型是:", type(ckpt))
