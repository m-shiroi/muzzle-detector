#!/usr/bin/env python


import io
import cv2
import torch
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def get_data(onTensorLoad):
    settings = {"interval": "1000", "count": "20"}
    url = "http://192.168.2.113:5000/stream"  # Note: change to ur ip

    r = requests.get(url, params=settings, stream=True)

    buffer = b""
    data_label = b"--frame"
    payload_label = b"--coords"
    payload_flag = False
    payload = None

    for line in r.iter_lines():

        if line == b"Content-Type: image/jpeg":
            buffer = b""
        elif payload_flag:
            payload = np.frombuffer(line, dtype=np.int32)
            payload_flag = False
        elif line == payload_label:
            payload_flag = True
        elif line == data_label:
            if not buffer:
                continue
            buff = io.BytesIO(buffer)

            try:
                tensor = torch.load(buff, encoding="latin1")
            except:
                print("[ERROR]: Corrupt frame, skipping.")
                continue

            onTensorLoad(tensor, payload)
        else:
            buffer += line + b"\n" if line else b""


def draw_rectangle(cv_img, coords, color=(255, 192, 203)):
    y, x, h, w, rOffset = coords
    cv2.rectangle(
        cv_img,
        (x + w + rOffset, y + h + rOffset),
        (x - rOffset, y - rOffset),
        color,
        2,
    )


# TODO: change func name
def test(tensor, coords):
    transformation = transforms.ToPILImage()
    img = transformation(tensor)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # print("tensor", "-- tensor test")
    # print(coords)

    # Note: here u can access full frame as 'img_cv' & detected face coordinates via 'coords'
    # Model will probably be here
    draw_rectangle(img_cv, coords, (0, 255, 0))

    # y, x, h, w, rOffset = coords
    # imgCrop = img[y:y+h+rOffset, x:x+w+rOffset].copy()

    cv2.imshow("image", img_cv)
    cv2.waitKey(1)


if __name__ == "__main__":
    get_data(test)
