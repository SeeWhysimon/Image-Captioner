import torch
from torchvision import transforms

# ========================== mode ==========================
mode = "evaluate"

# ========================== data ==========================
percent = 0.5
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# ========================== model ==========================
backbone = "resnet34"
embed_size = 256
hidden_size = 512
checkpoint_path = None

# ========================== train ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_every = 1
batch_size = 32
num_epochs = 50
learning_rate = 1e-3
scheduler_step_size = 100
padding_idx = 0

# ========================== evaluate ==========================

# ========================== refer ==========================
max_len = 20
images = "D:/Image_Captioning_COCO/coco/test2017/000000000019.jpg"

# ========================== result ==========================
save_every = 10
save_path = "../results/"