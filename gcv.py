import argparse
import io
import os
import random

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.utils as ch_utils
from scipy.fftpack import dct, idct
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm

import utils
# https://cloud.google.com/vision/docs/quickstart-client-libraries#client-libraries-install-python
from google.cloud import vision
from google.cloud.vision import types

parser = argparse.ArgumentParser()
parser.add_argument("--low-dim", type=int, default=1500)
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--num", type=int, default=50)
args = parser.parse_args()
print(args)

np.random.seed(5677)
random.seed(5677)
torch.manual_seed(5677)
torch.cuda.manual_seed(5677)
torch.cuda.manual_seed_all(5677)

LOW_DIM = args.low_dim
FREQ_DIM = 28
STRIDE = 7
MODEL = args.model
DATA_ROOT = "./imgs"
mom = args.mom

client = vision.ImageAnnotatorClient()


def normalize(x):
    return utils.apply_normalization(x, "imagenet")


N_query = 0


# Loss fuction for Google Cloud Vision
def f(x, y=None):
    global N_query
    N_query += x.shape[0]
    filename = "/tmp/google.bmp"
    ch_utils.save_image(x.clone(), filename)
    with io.open(filename, "rb") as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    response = client.label_detection(image=image)
    label_annotations = response.label_annotations
    labels = []
    scores = []
    for label in label_annotations:
        labels.append(label.description)
        scores.append(label.score)
    if y is None:
        max_tag = labels[:3]
        max_confidence = max(scores[:3])
        return max_tag, max_confidence
    max_confidence = 0
    for i in range(len(labels)):
        if labels[i] in y:
            max_confidence = max(max_confidence, scores[i])
    return max_confidence


def get_probability(success_probability):
    probability = [v / sum(success_probability) for v in success_probability]
    return probability


def PPBA(x, y, index, num_iter=40, epsilon=0.2):
    def func(xs):
        z = torch.from_numpy(xs).float().view(-1, LOW_DIM)
        perturbation = (
            (z @ Random_Matrix)
            .view(z.shape[0], 3, image_size, image_size)
            .clamp(-16 / 255, 16 / 255)  # Norm constraint
        )
        new_image = (x + perturbation).clamp(0, 1)
        loss = f(new_image, y)
        loss = np.array([loss])
        return loss

    global N_query
    variables = LOW_DIM
    step = 0.1
    z = np.zeros((1, LOW_DIM))
    y, prev_f = f(x)
    is_success = 0 if prev_f > 0 else 1
    success_number = [
        np.ones((1, LOW_DIM)),
        np.ones((1, LOW_DIM)),
        np.ones((1, LOW_DIM)),
    ]
    failed_number = [
        np.ones((1, LOW_DIM)),
        np.ones((1, LOW_DIM)),
        np.ones((1, LOW_DIM)),
    ]

    move_time = 0
    move_step = []
    move_func = []
    failed_time = 0
    success_norm = []
    fail_norm = []
    last_u = np.zeros((LOW_DIM,))

    worked = 0
    not_worked = 0

    for k in range(num_iter):
        u = np.zeros((1, LOW_DIM))
        r = np.random.uniform(size=(1, LOW_DIM))
        success_probability = [
            success_number[i] / (success_number[i] + failed_number[i])
            for i in range(len(success_number))
        ]
        probability = get_probability(success_probability)

        u[r < probability[0]] = -1
        u[r >= probability[0] + probability[1]] = 1

        uz = z + step * u

        fu = func(uz)
        print(fu)
        if fu.min() < prev_f:
            worked_u = u[fu < prev_f]
            success_number[0] = success_number[0] * mom + (worked_u == -1).sum(0)
            success_number[1] = success_number[1] * mom + (worked_u == 0).sum(0)
            success_number[2] = success_number[2] * mom + (worked_u == 1).sum(0)
            not_worked_u = u[fu >= prev_f]
            failed_number[0] = failed_number[0] * mom + (not_worked_u == -1).sum(0)
            failed_number[1] = failed_number[1] * mom + (not_worked_u == 0).sum(0)
            failed_number[2] = failed_number[2] * mom + (not_worked_u == 1).sum(0)
            z = uz[np.argmin(fu)]
            prev_f = fu.min()
        else:
            failed_number[0] += (u == -1).sum(0)
            failed_number[1] += (u == 0).sum(0)
            failed_number[2] += (u == 1).sum(0)
            failed_time += 1

        if prev_f <= 0:
            is_success = 1
            break

    current_q = N_query
    N_query = 0

    z = torch.from_numpy(z).float()
    perturbation = (z @ Random_Matrix).view(1, 3, image_size, image_size)
    new_image = (x + perturbation).clamp(0, 1)
    return current_q, is_success, perturbation.view(1, -1).norm(2, 1).item()


if MODEL.startswith("inception"):
    image_size = 299
    testset = dset.ImageFolder(DATA_ROOT, utils.INCEPTION_TRANSFORM)
else:
    image_size = 224
    testset = dset.ImageFolder(DATA_ROOT, utils.IMAGENET_TRANSFORM)

Random_Matrix = np.zeros((LOW_DIM, 3 * image_size * image_size))
indices = utils.block_order(image_size, 3, initial_size=FREQ_DIM, stride=STRIDE)
for i in range(LOW_DIM):
    Random_Matrix[i][indices[i]] = 1
Random_Matrix = (
    torch.from_numpy(
        idct(
            idct(
                Random_Matrix.reshape(-1, 3, image_size, image_size),
                axis=3,
                norm="ortho",
            ),
            axis=2,
            norm="ortho",
        )
    )
    .view(-1, 3, image_size, image_size)
    .float()
).view(LOW_DIM, -1)

# Attack the 1st image
image = testset[0][0].unsqueeze(0)
PPBA(image, None, 0)
