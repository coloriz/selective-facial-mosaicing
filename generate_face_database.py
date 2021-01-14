import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from threading import Thread

import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import cv2 as cv

from facealigner import FaceAligner
from helper import load_database, save_database, l2_normalize, standardize_image
from webcamreader import WebCamReader

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('db_path', type=Path, help='path to database file (.json)')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--camera-index', type=int, help='camera index to open')
group.add_argument('--glob', help='read all images given by this relative pattern')
parser.add_argument('--model-path', default='model/facenet_keras.h5', type=Path, help='path to facenet model')
opt = parser.parse_args()

detector = MTCNN()
model: tf.keras.Model = tf.keras.models.load_model(opt.model_path)
face_aligner = FaceAligner()

WINDOW_NAME = 'register'
SELECTED_FACE = 'selected'
param = [None, []]
aligned_face = None
identifier = None
get_input_thread = None

database = load_database(opt.db_path)


def get_input():
    global identifier, get_input_thread
    identifier = input('Enter ID: ')
    cv.destroyWindow(SELECTED_FACE)
    get_input_thread = None


def on_mouse_event(event, mx, my, flags, param):
    global aligned_face, get_input_thread
    if event != cv.EVENT_LBUTTONUP or get_input_thread is not None:
        return
    img, results = param
    for face in results:
        x, y, w, h = face['box']
        if x < mx < x + w and y < my < y + h:
            keypoints = face['keypoints']
            aligned_face = face_aligner.align(img, keypoints['left_eye'], keypoints['right_eye'])
            cv.imshow(SELECTED_FACE, aligned_face)
            get_input_thread = Thread(target=get_input, daemon=True)
            get_input_thread.start()
            return


cv.namedWindow(WINDOW_NAME)
cv.setMouseCallback(WINDOW_NAME, on_mouse_event, param)

if opt.glob:
    imgs = Path('.').glob(opt.glob)
    cap = map(lambda p: cv.imread(os.fspath(p)), imgs)
    delay = 0
else:
    cap = WebCamReader(opt.camera_index)
    cap.start()
    delay = 1

for img in cap:
    canvas = img.copy()
    results = detector.detect_faces(img[..., ::-1])
    param[0], param[1] = img, results
    if results:
        for face in results:
            x, y, w, h = face['box']
            cv.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)

    if identifier is not None and aligned_face is not None:
        # Preprocess face image before feeding into the model
        batch_faces = standardize_image(aligned_face[np.newaxis, ...].astype(np.float32))
        ov: np.ndarray = model.predict_on_batch(batch_faces)[0]
        if identifier in database.keys():
            print(f'The key "{identifier}" already exists in database. This will be replaced with the new one.')
        database[identifier] = l2_normalize(ov)
        cv.destroyWindow(SELECTED_FACE)
        identifier, aligned_face = None, None

    cv.imshow(WINDOW_NAME, canvas)
    if cv.waitKey(delay) & 0xFF == ord('q'):
        break

if opt.camera_index is not None:
    cap.stop()
save_database(opt.db_path, database)
