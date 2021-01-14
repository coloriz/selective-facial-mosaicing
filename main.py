import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import cv2 as cv

from facealigner import FaceAligner
from helper import put_text, load_database, l2_normalize, standardize_image

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('db_path', type=Path, help='path to database file (.json)')
parser.add_argument('key', help='database key of the face that would not be mosaiced')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--camera-index', type=int, help='camera index to open')
group.add_argument('--video-path', help='input video to read')
parser.add_argument('--model-path', default='model/facenet_keras.h5', type=Path, help='path to facenet model')
parser.add_argument('--threshold', default=0.8, type=float, help='threshold of euclidean distance between two vectors')
opt = parser.parse_args()

detector = MTCNN()
model: tf.keras.Model = tf.keras.models.load_model(opt.model_path)
face_aligner = FaceAligner()

database = load_database(opt.db_path)
ref_vector = database[opt.key]

if opt.video_path:
    cap = cv.VideoCapture(opt.video_path)
else:
    cap = cv.VideoCapture(opt.camera_index)


def extract_faces_and_standardize(img: np.ndarray):
    """
    Extract faces from the givin image and standardize them.
    :param img: OpenCV BGR image.
    :return: (batch of face images, detection results)
    """
    img = img[..., ::-1]
    results = detector.detect_faces(img)
    if not results:
        return None, None
    faces = map(lambda face: face_aligner.align(img, face['keypoints']['left_eye'], face['keypoints']['right_eye']), results)
    faces = np.array(list(faces), np.float32)

    # standardize pixel values across channels
    faces = standardize_image(faces)
    return faces, results


while True:
    ret, img = cap.read()
    if not ret:
        break
    faces, results = extract_faces_and_standardize(img)
    if faces is not None:
        outputs = model.predict_on_batch(faces)
        outputs = map(l2_normalize, outputs)
        for face, ov in zip(results, outputs):
            x, y, w, h = face['box']
            x, y = abs(x), abs(y)
            face_region = img[y:y+h, x:x+w]
            distance = np.linalg.norm(ref_vector - ov)
            rect_color = (0, 0, 255)
            if distance > opt.threshold:
                cv.blur(face_region, (11, 11), face_region)
                rect_color = (0, 255, 0)
            cv.rectangle(img, (x, y), (x + w, y + h), rect_color, 1)
            put_text(img, f'{distance:.3f}, {face["confidence"]:.2f}', (x, y - 15), (0, 255, 0))

    cv.imshow('video', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
