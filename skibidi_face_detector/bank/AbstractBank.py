from torch import Tensor
import pickle


Label = int | str


class AbstractBank:
    def __init__(self):
        self.embeddings = []
        self.labels = []

    def add_embedding(self, embedding: Tensor, label: Label) -> None:
        raise NotImplementedError

    def predict(self, embeddings: Tensor) -> list[Label]:
        raise NotImplementedError

    def prepare(self) -> None:
        """
        Not necessary, only for standardization of subclasses.
        """

    def save(self, file_path: str):
        with open(file_path, 'wb') as file:
            pickle.dump(self.__dict__, file)
        print(f"Bank's parameters saved successfully to {file_path}.")

    def load(self, file_path: str):
        with open(file_path, 'rb') as file:
            self.__dict__.update(pickle.load(file))
        self.prepare()
        print(f"Bank's parameters loaded successfully from {file_path}.")
