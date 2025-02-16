from .__utils import detect_faces, find_pose
from torch import Tensor


def get_quality(image: Tensor, face_detections=None):
    if face_detections is None:
        face_detections = detect_faces(image)

    assessment_parameters = []

    for face_detection in face_detections:
        confidence = face_detection["confidence"]
        head_position = find_pose(face_detection["keypoints"])

        assessment_parameters.append({'confidence': confidence, **head_position})

    return len(face_detections), assessment_parameters


def assess_quality(image: Tensor, face_detections=None, *, min_confidence: float = 0.99, single_face_only: bool = True, max_yaw: int = 30, max_pitch: int = 60):
    n_faces, faces_quality = get_quality(image, face_detections)

    if n_faces == 0:
        return False, 'No faces found', [], [n_faces, faces_quality]

    if single_face_only and n_faces > 1:
        return False, f'More than one face found when {single_face_only = }', [], [n_faces, faces_quality]

    faces_acceptances = []
    for face_quality in faces_quality:
        if face_quality["confidence"] < min_confidence:
            faces_acceptances.append((False, f'{face_quality["confidence"] = } < {min_confidence = }'))
        elif abs(face_quality["yaw"]) > max_yaw:
            faces_acceptances.append((False, f'{abs(face_quality["yaw"]) = } > {max_yaw = }'))
        elif abs(face_quality["pitch"]) > max_pitch:
            faces_acceptances.append((False, f'{abs(face_quality["pitch"]) = } > {max_pitch = }'))
        else:
            faces_acceptances.append((True, 'All conditions satisfied'))

    return True, 'All conditions satisfied', faces_acceptances, [n_faces, faces_quality]
