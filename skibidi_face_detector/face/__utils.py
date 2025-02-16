from torch import Tensor
from mtcnn import MTCNN
import einops
import numpy as np
import torch
import warnings
from typing import Literal

detector = MTCNN()
HeadPoseAngle = Literal['roll', 'pitch', 'yaw']


def tensor2cv(tensor: Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = einops.rearrange(tensor, '1 c h w -> c h w')

    return einops.rearrange(tensor, 'c h w -> h w c').numpy()


def cv2tensor(cv: np.ndarray) -> Tensor:
    return torch.from_numpy(einops.rearrange(cv, 'h w c -> c h w'))


def detect_faces(image):
    assert isinstance(image, Tensor), 'Should pass a tensor.'
    assert image.ndim in [3, 4], 'Should pass a tensor with 3 or 4 dims: (batch), color, height, width.'

    assert image.shape[0] == 3 or image.shape[1] == 3, 'Should pass an RGB image with dims: (batch), color, height, width.'

    detection_ready_image = tensor2cv(image) * 255

    return detector.detect_faces(detection_ready_image)


def detect_face(image, *, warn=True):
    face_detections = detect_faces(image)

    n_faces = len(face_detections)

    if n_faces == 0:
        raise ValueError("No faces detected.")

    if n_faces != 1:
        if warn:
            warnings.warn(f"Found {n_faces} faces. Aligning to the one with biggest confidence.")
        face_detection = max(face_detections, key=lambda fd: fd['confidence'])
    else:
        face_detection = face_detections[0]

    return face_detection


def find_pose(keypoints) -> dict[HeadPoseAngle, float]:
    left_eye_x, left_eye_y = keypoints["left_eye"]
    right_eye_x, right_eye_y = keypoints["right_eye"]

    dPx_eyes = max((right_eye_x - left_eye_x), 1)
    dPy_eyes = (right_eye_y - left_eye_y)

    roll = np.arctan(dPy_eyes / dPx_eyes)

    cos_theta = np.cos(roll)
    sin_theta = np.sin(roll)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    rotated_keypoints = {}
    for key, (x, y) in keypoints.items():
        point = np.array([x, y]) - np.array([left_eye_x, left_eye_y])
        rotated_point = np.dot(rotation_matrix, point) + np.array([left_eye_x, left_eye_y])
        rotated_keypoints[key] = (rotated_point[0], rotated_point[1])

    rotated_eye_right, rotated_eye_left = rotated_keypoints["right_eye"], rotated_keypoints["left_eye"]
    rotated_mouth_right, rotated_mouth_left = rotated_keypoints["mouth_right"], rotated_keypoints["mouth_left"]
    rotated_nose = rotated_keypoints["nose"]

    dXtot = (rotated_eye_right[0] - rotated_eye_left[0] + rotated_mouth_right[0] - rotated_mouth_left[0]) / 2
    dYtot = (rotated_mouth_left[1] - rotated_eye_left[1] + rotated_mouth_right[1] - rotated_eye_right[1]) / 2

    dXnose = (rotated_eye_right[0] - rotated_nose[0] + rotated_mouth_right[0] - rotated_nose[0]) / 2
    dYnose = (rotated_mouth_left[1] - rotated_nose[1] + rotated_mouth_right[1] - rotated_nose[1]) / 2

    yaw = (-90 + 90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
    pitch = (-90 + 90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0

    return {'roll': roll * 180 / np.pi, 'pitch': pitch, 'yaw': -yaw}
