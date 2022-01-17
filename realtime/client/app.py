#!/usr/bin/env python

import requests
from PIL import Image
import io
import numpy as np
import torch
import cv2


def get_data(onTensorLoad):
    settings = {"interval": "1000", "count": "20"}
    url = "http://192.168.2.113:5000/stream"

    r = requests.get(url, params=settings, stream=True)

    buffer = b""
    delim = b"--frame"
    for line in r.iter_lines():
        if line == b"Content-Type: image/jpeg":
            buffer = b""

        elif line == delim:
            if not buffer:
                continue

            buff = io.BytesIO(buffer)
            tensor = torch.load(buff, encoding="latin1")
            onTensorLoad(tensor)

        else:
            buffer += line + b"\n" if line else b""


def test(tensor):
    print(tensor, "-- tensor test")


if __name__ == "__main__":
    get_data(test)
