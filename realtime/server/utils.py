import sys
import cv2
import io
import numpy
import face_recognition as fr
import matplotlib.pyplot as plt
import torch
import io
from PIL import Image
import torchvision.transforms as transforms


class Stream:
    def __init__(self):
        self.classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_scale = 20
        self.video_source = -1

    def get_frames(self):
        video_cap = cv2.VideoCapture(self.video_source)
        if not video_cap.isOpened():
            raise RuntimeError("Unable open video capture")

        while True:
            _, img = video_cap.read()

            for coords in self.locate_faces(img):
                x, y, w, h = coords

                img_crop = img[
                    y : y + h + self.face_scale, x : x + w + self.face_scale
                ].copy()
                img_pil = Image.fromarray(img_crop)

                eval_transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                tensor = eval_transform(img_pil)

                buff = io.BytesIO()
                torch.save(tensor, buff)
                buff.seek(0)

                yield buff.read()  # <-- passing torch tensor
                # yield cv2.imencode(".jpg", img_crop)[1].tobytes() # <-- to pass raw image

            if cv2.waitKey(1) & 0xFF == ord("q"):
                video_cap.release()
                break

    def locate_faces(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.classifier.detectMultiScale(img_gray, 1.1, 4)

    def gen(self):
        yield b"--frame\r\n"

        frame_gen = self.get_frames()
        while True:
            frame = next(frame_gen)
            yield b'Content-Type: image/jpeg\n\n' + frame + b'\n--frame\n'
