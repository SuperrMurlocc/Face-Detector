from torch import Tensor
from .__utils import tensor2cv, cv2tensor, detect_face, detect_faces
from .__settings import DESIRED_FACE_WIDTH, DESIRED_RATIO, DESIRED_FACE_HEIGHT
import cv2
import math
import numpy as np


def _align_eyes(image: Tensor, face_detection) -> (Tensor, tuple[int, int, int, int]):
    cv2image = tensor2cv(image)

    left_eye_x, left_eye_y = face_detection["keypoints"]["left_eye"]
    right_eye_x, right_eye_y = face_detection["keypoints"]["right_eye"]

    mid_eye_x, mid_eye_y = (left_eye_x + right_eye_x) / 2, (left_eye_y + right_eye_y) / 2

    dx = right_eye_x - left_eye_x
    dy = left_eye_y - right_eye_y
    angle = -math.degrees(math.atan2(dy, dx))

    M = cv2.getRotationMatrix2D([mid_eye_x, mid_eye_y], angle, 1.0)
    rotated_image = cv2.warpAffine(cv2image, M, (image.shape[-1], image.shape[-2]))

    x, y, w, h = face_detection["box"]

    corners = np.array([[x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]], dtype=np.float32)

    rotated_corners = cv2.transform(np.array([corners]), M)[0]

    x = x_min = np.min(rotated_corners[:, 0])
    y = y_min = np.min(rotated_corners[:, 1])
    x_max = np.max(rotated_corners[:, 0])
    y_max = np.max(rotated_corners[:, 1])
    w = x_max - x_min
    h = y_max - y_min

    x, y, w, h = _convert_box_to_desired_ratio(x, y, w, h)

    x = max(0, x)
    y = max(0, y)

    return cv2tensor(rotated_image), (x, y, w, h)


def _convert_box_to_desired_ratio(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    expected_h = w * DESIRED_RATIO
    if h > expected_h:
        w_diff = h / DESIRED_RATIO - w

        x -= w_diff / 2
        w += w_diff
    else:
        h_diff = expected_h - h

        y -= h_diff / 2
        h += h_diff

    return int(x), int(y), int(w), int(h)


def _resize_to_desired_width_and_height(face_image: Tensor) -> Tensor:
    return cv2tensor(cv2.resize(tensor2cv(face_image), (DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT)))


def _align_face(image: Tensor, face_detection):
    rotated_image, (x, y, w, h) = _align_eyes(image, face_detection)

    face_image = rotated_image[..., y:y + h, x:x + w]

    face_image = _resize_to_desired_width_and_height(face_image)

    return face_image


def align_most_confident_face(image: Tensor, *, warn: bool = True) -> Tensor:
    face_detection = detect_face(image, warn=warn)

    return _align_face(image, face_detection)


def align_faces(image: Tensor, face_detections=None) -> list[Tensor]:
    if face_detections is None:
        face_detections = detect_faces(image)

    face_images = []
    for face_detection in face_detections:
        face_image = _align_face(image, face_detection)
        face_images.append(face_image)

    return face_images