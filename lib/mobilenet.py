import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MobileKeypointNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(
            weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1"
        )
        # Заменяем последние 2 полносвязных слоя на имеющие нужную размерность
        self.backbone.classifier[0] = nn.Linear(576, 256)
        self.backbone.classifier[3] = nn.Linear(256, 8)
        
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=4)
        self.fc = nn.Linear(in_features=12, out_features=2)
        self.bn = nn.BatchNorm1d(num_features=12, affine=True)
        
        
    def get_layers_names(self):
        return dict(self.backbone.named_modules())

    def forward(self, image, dataset_idx):
        image_embedding = self.backbone(image)
        dataset_embedding = self.embedding(dataset_idx)
        full_embedding = torch.hstack((image_embedding, dataset_embedding))
        out = F.relu(full_embedding)
        out = self.bn(out)
        out = self.fc(out)
        
        return out
