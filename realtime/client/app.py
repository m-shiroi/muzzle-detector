#!/usr/bin/env python


import io
import cv2
import torch
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
from statistics import mode
from torch.utils.data import DataLoader
from torch import nn

CORRECT = 0
INCORRECT = 1

GREEN = (0, 255, 0)
RED = (0, 0, 255)

def get_data(onTensorLoad):
    settings = {"interval": "1000", "count": "20"}
    url = "http://192.168.88.154:5000/stream"  # Note: change to ur ip

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

def predict(tns):
    with torch.no_grad():
        if len(tns.size()) == 3:
            data = tns.unsqueeze(0)
        else:
            data = tns
        output = model(data)
        ret, prediction = torch.max(output.data, 1)
        
        return mode ( prediction )

# TODO: change func name
def test(tensor, coords):
    transformation = transforms.ToPILImage()
    trf= transforms.Compose([
            transformation,
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
    ])
    
    
    img = transformation(tensor)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # print("tensor", "-- tensor test")
    # print(coords)

    # Note: here u can access full frame as 'img_cv' & detected face coordinates via 'coords'
    # Model will probably be here
    
    crop = lambda tsr, to_data, y, x, h, w, _: to_data(
                tsr[:, y: y+h, x: x+w]
            )  
    

    
    res = predict ( crop ( tensor, trf, *coords ) )
    draw_rectangle(img_cv, coords, GREEN if res == CORRECT else RED)
    
    
    # y, x, h, w, rOffset = coords
    # imgCrop = img[y:y+h+rOffset, x:x+w+rOffset].copy()

    cv2.imshow("image", img_cv)
    cv2.waitKey(1)

RES32 = "../../models/resnet34_SMALL_10_epoch_Last_Linear.pt"
MOBILE = "../../models/mobilenet_v3_small_1_Linear_25e_20000_7000.pt"

if __name__ == "__main__":
    my_model = MOBILE
    if my_model == RES32:
        model = torchvision.models.resnet34()
        model.fc = nn.Linear(512, 2)
    elif my_model == MOBILE:
        model = torchvision.models.mobilenet_v3_small()
        model.fc = nn.Linear(1024, 2)
        
    model.load_state_dict(torch.load(my_model))
    model.eval()
    get_data(test)
