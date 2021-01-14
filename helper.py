from enum import IntEnum, auto
import json
from pathlib import Path

import cv2 as cv
import numpy as np


class Alignment(IntEnum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


def put_text(img, text, position, color, alignment=Alignment.LEFT, scale=None, thickness=None, line_type=cv.LINE_8):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = scale or 0.8
    font_thickness = thickness or 2
    shadow_offset = 2
    text_size, baseline = cv.getTextSize(text, font, font_scale, font_thickness)
    if alignment == Alignment.LEFT:
        final_position = (position[0], position[1] + text_size[1] // 2)
    elif alignment == Alignment.CENTER:
        final_position = (position[0] - text_size[0] // 2, position[1] + text_size[1] // 2)
    else:
        final_position = (position[0] - text_size[0], position[1] + text_size[1] // 2)

    cv.putText(img, text, (final_position[0] + shadow_offset, final_position[1] + shadow_offset),
               font, font_scale, (0, 0, 0), font_thickness, line_type)
    cv.putText(img, text, final_position, font, font_scale, color, font_thickness, line_type)


def standardize_image(img: np.ndarray):
    """img: 4-dim tensor. (B, H, W, C)"""
    mean = img.mean((1, 2, 3), keepdims=True)
    std = img.std((1, 2, 3), keepdims=True)
    return (img - mean) / std


def load_database(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open('r') as f:
        db: dict = json.load(f)
    db = {k: np.array(v, np.float32) for k, v in db.items()}
    return db


def save_database(path: str, db: dict) -> None:
    database = Path(path)
    db_for_save = {k: v.tolist() for k, v in db.items()}
    with database.open('w') as f:
        json.dump(db_for_save, f)


def l2_normalize(x: np.ndarray):
    return x / np.linalg.norm(x)
