from ..bank import KnnBank
from ..face_embedder.Model import Model
from torch.utils.data import DataLoader
from torch import Tensor
import torch
from PIL import Image
from torchvision.transforms import v2 as transforms
from ..face import align_faces, detect_faces, assess_quality
import numpy as np
import cv2
import einops
from tqdm import tqdm
import matplotlib.pyplot as plt


def batched(tensor: Tensor) -> Tensor:
    return torch.unsqueeze(tensor, dim=0)


def unbatched(tensor: Tensor) -> Tensor:
    return torch.squeeze(tensor)


class App:
    def __init__(self, *, bank_file: str = None, **kwargs):
        self.bank = KnnBank(**kwargs)

        if bank_file is not None:
            self.bank.load(bank_file)

        self.model = None

    def load_model(self, model_file: str, **kwargs):
        self.model = Model.load_from_checkpoint(model_file, **kwargs)
        self.model.augments = None
        self.model.transformer = None
        self.model.eval()

    @torch.inference_mode()
    def load_dataset(self, loader: DataLoader, *, idx_to_class: dict[int, str] = None):
        assert self.model is not None, "Model not loaded, please run .load_model()"

        self.bank.idx_to_class = idx_to_class

        X = []
        Y = []

        for batch in tqdm(loader, 'Loading Dataset'):
            x, y = self.model.transform_batch({'image': batch[0], 'label': batch[1]})

            # plt.imshow(einops.rearrange(x, '1 c h w -> h w c').cpu().numpy())
            # plt.show()

            if len(x) == 0:
                continue

            embeddings = self.model(x)

            X.append(embeddings.cpu())
            Y.append(y.cpu())

        Xs = torch.cat(X)
        Ys = torch.cat(Y)

        for embedding, label in zip(Xs, Ys):
            self.bank.add_embedding(embedding, label)

        self.bank.prepare()

    def assign_image_from_file(self, image_file: str):
        image = Image.open(image_file).convert("RGB")

        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])

        tensor_image = transform(image)

        return self.assign_image(tensor_image)

    @staticmethod
    def convert2image(image, face_detections, predicts):
        numpy_image = np.array(einops.rearrange(image, 'c h w -> h w c'))

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1
        font_thickness = 1
        text_color = (1.0, 1.0, 1.0)

        color = (1.0, 0.0, 0.0)
        thickness = 3

        for face_detection, predict in zip(face_detections, predicts):
            x, y, w, h = face_detection['box']
            label = predict

            (_, _), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_x = x
            label_y = y

            cv2.rectangle(numpy_image, (x, y), (x + w, y + h), color, thickness)

            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            text_w, text_h = text_size
            cv2.rectangle(numpy_image, (label_x, label_y), (x + text_w, y + text_h + 1), color, -1)
            cv2.putText(numpy_image, label, (label_x, label_y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

        return numpy_image

    @torch.inference_mode()
    def assign_image(self, image: Tensor):
        face_detections = detect_faces(image)
        faces_images = []

        image_ok, image_message, faces_ok_and_messages, qualities = assess_quality(image, face_detections, single_face_only=False, min_confidence=0.95, max_yaw=40)
        if not image_ok:
            print(image_message)
        else:
            for face_image, (face_ok, face_message) in zip(align_faces(image, face_detections), faces_ok_and_messages):
                if not face_ok:
                    faces_images.append(None)
                else:
                    faces_images.append(face_image)

        if len(faces_images) == 0:
            return None

        embeddings = []
        for face_image in faces_images:
            if face_image is not None:
                embedding = self.model(batched(face_image).to(self.model.device)).cpu()
            else:
                embedding = None
            embeddings.append(embedding)

        predicts = []
        for embedding in embeddings:
            if embedding is not None:
                predict = self.bank.predict(embedding)[0]
            else:
                predict = 'not a proper face'
            predicts.append(predict)

        image = self.convert2image(image, face_detections, predicts)

        return image


