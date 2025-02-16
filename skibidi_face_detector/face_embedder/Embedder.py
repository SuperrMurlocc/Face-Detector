from torch import nn
from .VggFace2 import vgg_face_2
from .utils import load_state_dict
import os


class Embedder(nn.Module):
    def __init__(self, *, embedding_dim: int = 100, hidden_layer_features: int = 1024, p_dropout_1: float = 0.25, p_dropout_2: float = 0.25, freeze_feature_extractor: bool = False):
        super().__init__()

        backbone = vgg_face_2(include_top=False)
        load_state_dict(backbone, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained', 'resnet50_ft_weight.pkl'))
        self.feature_extractor = nn.Sequential(
            backbone,
            nn.Flatten()
        )

        if freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.embedder = nn.Sequential(
            nn.Dropout(p=p_dropout_1),
            nn.Linear(in_features=2048, out_features=hidden_layer_features),
            nn.ReLU(),
            nn.Dropout(p=p_dropout_2),
            nn.Linear(in_features=hidden_layer_features, out_features=embedding_dim),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        embeddings = self.embedder(features)
        return embeddings
