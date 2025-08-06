import torch.nn as nn
import torchvision.models as models
import torch

class CNNEncoder(nn.Module):
    def __init__(self, embed_size, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.feature_dim = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        for param in base.parameters():
            param.requires_grad = False

        for param in base.layer3.parameters():
            param.requires_grad = True
        for param in base.layer4.parameters():
            param.requires_grad = True
            
        self.backbone = nn.Sequential(*list(base.children())[:-1]) 
        self.classifier = nn.Sequential(nn.Linear(self.feature_dim, embed_size * 2),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.3),
                                        nn.Linear(embed_size * 2, embed_size),
                                        nn.LayerNorm(embed_size))

    def forward(self, images):
        features = self.backbone(images)
        features = features.view(features.size(0), -1)
        features = self.classifier(features)
        return features

class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # captions: [B, T] with <SOS> ... <EOS>
        embeddings = self.embedding(captions[:, :-1])  # 去掉最后一个<EOS>
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # [B, T+1, E]
        outputs, _ = self.rnn(inputs)
        outputs = self.fc(outputs)  # [B, T+1, vocab_size]
        return outputs[:, 1:, :]  # 丢掉第一个output，只保留和target对齐的部分