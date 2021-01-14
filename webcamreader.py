from threading import Thread

import cv2 as cv
import numpy as np


class WebCamReader(Thread):
    def __init__(self, index=0):
        Thread.__init__(self)
        self.cap = cv.VideoCapture(index)
        self.grabbed, self.frame = self.cap.read()
        self.is_running = False

    def run(self) -> None:
        while self.is_running:
            self.grabbed, self.frame = self.cap.read()

    def read(self) -> np.ndarray:
        return self.frame

    def __iter__(self):
        return self

    def __next__(self):
        if not self.grabbed:
            raise StopIteration
        return self.frame

    def start(self) -> None:
        self.is_running = True
        super().start()

    def stop(self) -> None:
        self.is_running = False
        self.join()
