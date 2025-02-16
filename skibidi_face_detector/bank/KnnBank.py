from numpy.linalg import norm
from numpy import dot, sqrt
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor
import umap
import matplotlib.pyplot as plt
import torch
from .AbstractBank import AbstractBank, Label


class KnnBank(AbstractBank):
    @staticmethod
    def cosine_distance(a, b):
        cosine_similarity = dot(a, b) / (norm(a) * norm(b))
        return 1 - cosine_similarity

    @staticmethod
    def normalized_euclidean_distance(a, b):
        euclidean_distance = norm(a - b)
        max_distance = sqrt(len(a))
        normalized_distance = euclidean_distance / max_distance

        return normalized_distance

    def combined_distance(self, a, b):
        return self.cosine_distance(a, b) + self.normalized_euclidean_distance(a, b)

    def __init__(self, *, threshold=1.0, idx_to_class: dict[int, str] = None):
        super().__init__()
        self.threshold = threshold

        self.idx_to_class = idx_to_class
        self.knn = None

    def prepare(self):
        self.knn = KNeighborsClassifier(n_neighbors=1, metric=self.cosine_distance)
        self.knn.fit(self.embeddings, self.labels)

    def add_embedding(self, embedding: Tensor, label: Label):
        self.embeddings.append(embedding)
        self.labels.append(label)
        self.knn = None

    def present(self):
        umap_model_2d = umap.UMAP(n_components=2)
        latent_2d = umap_model_2d.fit_transform(self.embeddings)

        for label in set([int(label) for label in self.labels]):
            plt.scatter(latent_2d[torch.tensor(self.labels) == label, 0], latent_2d[torch.tensor(self.labels) == label, 1], alpha=.75, label=self.idx_to_class[label])
        plt.legend()
        plt.xlabel("UMAP Dim 1")
        plt.ylabel("UMAP Dim 2")
        plt.title("UMAP Visualization of Latent Space")
        plt.show()

    def predict(self, embeddings: Tensor) -> list[Label]:
        assert self.knn is not None, "Bank is not prepared for prediction, please run .prepare()"

        predicts = self.knn.predict(embeddings)
        distances, *_ = self.knn.kneighbors(embeddings)

        if self.idx_to_class is not None:
            predicts = ["unknown" if distance > self.threshold else f'{self.idx_to_class[predict]}: {round(1 - float(distance[0]), 3)}' for predict, distance in zip(predicts, distances)]
        else:
            predicts = ["unknown" if distance > self.threshold else f'{predict}: {round(1 - float(distance[0]), 3)}' for predict, distance in zip(predicts, distances)]

        return predicts

        # return [f'{self.idx_to_class[predict]}' for predict in predicts]
