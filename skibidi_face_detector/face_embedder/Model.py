import pytorch_lightning as pl
import torch
from torchvision.transforms.v2 import Compose
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .ArcFaceLoss import ArcFaceLoss
from .Embedder import Embedder
from .TripletLoss import TripletLoss
from .. import face
from .utils import get_accuracy


class Model(pl.LightningModule):
    def __init__(self,
                 num_classes: int, *,
                 embedding_dim: int = 100,
                 arc_face_margin: float = 0.3,
                 triplet_margin: float = 1.0,
                 scale: float = 30.0,
                 learning_rate: float = 1e-4,
                 hidden_layer_features: int = 1024,
                 p_dropout_1: float = 0.25,
                 p_dropout_2: float = 0.25,
                 freeze_feature_extractor: bool = False,
                 augments: Compose = None,
                 transformer: Compose = None,
                 accuracy_loaders: tuple[DataLoader, DataLoader] = None,
                 arc_face_loss_multiplier: float = 1.0,
                 triplet_loss_multiplier: float = 1.0,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.arc_face_margin = arc_face_margin
        self.triplet_margin = triplet_margin
        self.learning_rate = learning_rate
        self.hidden_layer_features = hidden_layer_features
        self.p_dropout_1 = p_dropout_1
        self.p_dropout_2 = p_dropout_2
        self.freeze_feature_extractor = freeze_feature_extractor
        self.arc_face_loss_multiplier = arc_face_loss_multiplier
        self.triplet_loss_multiplier = triplet_loss_multiplier

        self.embedder = Embedder(embedding_dim=self.embedding_dim, hidden_layer_features=self.hidden_layer_features, p_dropout_1=self.p_dropout_1, p_dropout_2=self.p_dropout_2, freeze_feature_extractor=self.freeze_feature_extractor)

        self.arc_face_loss = ArcFaceLoss(
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            margin=self.arc_face_margin,
            scale=self.scale
        )

        self.triplet_loss = TripletLoss(self.triplet_margin)

        self.accuracy_loaders = accuracy_loaders
        self.augments = augments
        self.transformer = transformer

        self.save_hyperparameters(ignore=['augments', 'transformer', 'accuracy_loaders'])

    def forward(self, x):
        return self.embedder(x)

    def transform_batch(self, batch):
        x, y = batch['image'], batch['label']

        if self.transformer is not None:
            x = self.transformer(x)

        x_cpu = x.cpu()
        y_cpu = y.cpu()

        face_detects = [face.detect_faces(_x_cpu) for _x_cpu in x_cpu]
        face_qualities = [face.assess_quality(_x_cpu, face_detect, single_face_only=False) for _x_cpu, face_detect in zip(x_cpu, face_detects)]
        idxs_ok = list(filter(lambda idx: face_qualities[idx][0] and face_qualities[idx][2][0], range(len(x_cpu))))

        if len(idxs_ok) == 0:
            return [], []

        aligned_faces = [face.align_faces(x_cpu[idx], face_detects[idx])[0] for idx in idxs_ok]

        if self.augments is not None:
            aligned_faces = self.augments(aligned_faces)

        x = torch.stack(aligned_faces).to(self.device)
        y = torch.stack([y_cpu[idx] for idx in idxs_ok]).to(self.device)

        return x, y

    def sample_triplets(self, embeddings, labels, margin=1.0):
        """
        Sample triplets for triplet loss.

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim) containing the embeddings.
            labels: Tensor of shape (batch_size) containing the class labels.
            margin: Margin for triplet loss (used for semi-hard negative mining).

        Returns:
            anchor: Tensor of shape (batch_size, embedding_dim).
            positive: Tensor of shape (batch_size, embedding_dim).
            negative: Tensor of shape (batch_size, embedding_dim).
        """
        batch_size = embeddings.size(0)

        pairwise_dist = F.pairwise_distance(embeddings.unsqueeze(1), embeddings.unsqueeze(0), p=2)

        anchors = []
        positives = []
        negatives = []

        for i in range(batch_size):
            anchor = embeddings[i]
            label = labels[i]

            positive_indices = torch.where(labels == label)[0]
            positive_indices = positive_indices[positive_indices != i]
            if len(positive_indices) == 0:
                continue

            positive_idx = positive_indices[torch.randint(0, len(positive_indices), (1,))]
            positive = embeddings[positive_idx]

            negative_indices = torch.where(labels != label)[0]

            anchor_positive_dist = pairwise_dist[i, positive_idx]
            negative_dists = pairwise_dist[i, negative_indices]

            semi_hard_negatives = negative_indices[
                (negative_dists > anchor_positive_dist) & (negative_dists < anchor_positive_dist + margin)]

            if len(semi_hard_negatives) > 0:
                negative_idx = semi_hard_negatives[torch.randint(0, len(semi_hard_negatives), (1,))]
            else:
                negative_idx = negative_indices[torch.randint(0, len(negative_indices), (1,))]

            negative = embeddings[negative_idx]

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        if len(anchors) == 0:
            return None, None, None

        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        return anchors, positives, negatives

    def step(self, batch, batch_idx, *, step_name='train'):
        x, y = self.transform_batch(batch)

        if len(x) == 0:
            return None

        embeddings = self.embedder(x)
        arc_face_loss = self.arc_face_loss(embeddings, y)

        anchor, positive, negative = self.sample_triplets(embeddings, y)
        if anchor is None:
            return None
        triplet_loss = self.triplet_loss(anchor, positive, negative)

        combined_loss = self.arc_face_loss_multiplier * arc_face_loss + self.triplet_loss_multiplier * triplet_loss

        self.log(f'{step_name}_arc_face_loss', arc_face_loss, on_epoch=True, prog_bar=True)
        self.log(f'{step_name}_triplet_loss', triplet_loss, on_epoch=True, prog_bar=True)
        self.log(f'{step_name}_loss', combined_loss, on_epoch=True, prog_bar=True)

        return combined_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, step_name='train')

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, step_name='val')

    def on_validation_epoch_end(self) -> None:
        if self.accuracy_loaders is None:
            return

        augments = self.augments
        transformer = self.transformer

        acc = get_accuracy(self, self.accuracy_loaders[0], self.accuracy_loaders[1])
        self.log('val_accuracy', acc, on_epoch=True, prog_bar=True)

        self.augments = augments
        self.transformer = transformer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
