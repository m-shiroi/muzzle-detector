import io
import sys

import cv2
import face_recognition as fr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class Stream:
    def __init__(self):
        self.classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_scale = 20
        self.video_source = -1

    def _convert_data_to_bytes(self, payload, img):
        nparr = np.array(
            payload,
            dtype=np.int32,
        )

        transform_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        img_pil = Image.fromarray(img)
        tensor = transform_to_tensor(img_pil)

        buff = io.BytesIO()
        torch.save(tensor, buff)
        buff.seek(0)
        return nparr.tobytes(), buff.read()

    def _locate_faces(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.classifier.detectMultiScale(img_gray, 1.1, 4)

    def _get_frames(self):
        video_cap = cv2.VideoCapture(self.video_source)
        if not video_cap.isOpened():
            raise RuntimeError("Unable open video capture")

        transfrom_normalize = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )

        while True:
            _, img = video_cap.read()
            img_pil = Image.fromarray(img)
            img = cv2.cvtColor(
                np.array(transfrom_normalize(img_pil)), cv2.COLOR_RGB2BGR
            )
            faces = self._locate_faces(img)

            if len(faces):
                for coords in faces:
                    x, y, w, h = coords
                    payload, frame = self._convert_data_to_bytes(
                        [y, x, h, w, self.face_scale], img
                    )
                    yield b"\n--coords\n" + payload + b"\n" + frame
            else:
                payload, frame = self._convert_data_to_bytes([], img)
                yield b"\n--coords\n" + payload + b"\n" + frame

            if cv2.waitKey(1) & 0xFF == ord("q"):
                video_cap.release()
                break

    def gen(self):
        yield b"--frame\r\n"

        frame_gen = self._get_frames()
        while True:
            frame = next(frame_gen)
            yield b"Content-Type: image/jpeg\n" + frame + b"\n--frame\n"
