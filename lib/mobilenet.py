import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F


class MobileKeypointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(
            weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1"
        )
        # Заменяем последние 2 полносвязных слоя на имеющие нужную размерность
        self.backbone.classifier[0] = nn.Linear(576, 256)
        self.backbone.classifier[3] = nn.Linear(256, 8)
        
        self.embedding = nn.Embedding(num_embeddings=7, embedding_dim=2)
        self.fc = nn.Linear(in_features=10, out_features=2)
        self.bn = nn.BatchNorm1d(num_features=10, affine=True)
        
        
    def get_layers_names(self):
        return dict(self.backbone.named_modules())

    def forward(self, image, dataset_idx):
        image_embedding = self.backbone(image)
        dataset_embedding = self.embedding(dataset_idx)
        full_embedding = torch.hstack((image_embedding, dataset_embedding))
        full_embedding = F.relu(full_embedding)
        full_embedding = self.bn(full_embedding)
        full_embedding = self.fc(full_embedding)
        
        return full_embedding
