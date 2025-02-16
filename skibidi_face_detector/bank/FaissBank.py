import faiss
import torch
from torch import Tensor
from .AbstractBank import AbstractBank, Label
import einops
from typing import Literal


def batched(x: Tensor) -> Tensor:
    return einops.rearrange(x, "d -> 1 d")


class FaissBank(AbstractBank):
    def __init__(self, *, embedding_dim: int = None, mode: Literal['cosine', 'L2'] = 'cosine'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.index = None

        if self.embedding_dim is not None:
            self.prepare()

    def prepare(self) -> None:
        if self.mode == 'cosine':
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.mode == 'L2':
            self.index = faiss.IndexFlatL2(self.embedding_dim)

    @staticmethod
    def _normalize_embeddings(embeddings: Tensor) -> Tensor:
        norms = torch.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def add_embedding(self, embedding: Tensor, label: Label) -> None:
        assert self.index is not None, "The index is not initialized."
        b_embedding = batched(embedding)
        normalized_b_embedding = self._normalize_embeddings(b_embedding)
        self.index.add(normalized_b_embedding)

        self.embeddings.append(embedding)
        self.labels.append(label)

    def predict(self, embeddings: Tensor) -> list[Label]:
        assert self.index is not None, "The index is not initialized."
        normalized_embeddings = self._normalize_embeddings(embeddings)
        distances, indices = self.index.search(normalized_embeddings, k=1)

        return [self.labels[i[0]] if i[0] != -1 else 'unknown' for i in indices]

    def save(self, file_path: str):
        faiss.write_index(self.index, file_path + ".faiss")
        super().save(file_path + ".pckl")
        print(f"Bank's FAISS index and metadata saved successfully.")

    def load(self, file_path: str):
        super().load(file_path + ".pckl")
        self.prepare()
        self.index = faiss.read_index(file_path + ".faiss")
        print(f"Bank's FAISS index and metadata loaded successfully.")


if __name__ == '__main__':
    emb1 = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,])
    emb2 = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,])

    bank = FaissBank(embedding_dim=8)

    bank.add_embedding(emb1, 1)
    bank.add_embedding(emb2, 2)

    bank.save('test')

    bank = FaissBank()
    bank.load('test')

    emb3 = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, ])
    emb4 = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ])

    import numpy as np
    print(bank.predict(torch.from_numpy(np.array([emb3, emb4]))))
