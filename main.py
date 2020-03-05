import argparse
import os
import random

import numpy as np
import torch
import torchvision.datasets as dset
from scipy.fftpack import dct, idct
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--low-dim", type=int, default=1500)
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--num", type=int, default=1000)
parser.add_argument("--mom", type=float, default=1)
parser.add_argument("--data_root", type=str, default="images")
parser.add_argument("--order", type=str, default="strided")
parser.add_argument("--r", type=int, default=2352)
parser.add_argument("--max_iter", type=int, default=2000)
parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--rho", type=float, default=0.01)
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
DATA_ROOT = args.data_root
MOM = args.mom
ORDER = args.order
R = args.r
MAX_ITER = args.max_iter
N_SAMPLES = args.n_samples
RHO = args.rho

if MODEL == "inception_v3":
    FREQ_DIM = 38
    STRIDE = 9


def normalize(x):
    return utils.apply_normalization(x, "imagenet")


N_query = 0


def cw_loss(x, y, targeted=False):
    global N_query
    N_query += x.shape[0]
    outputs = model(normalize(x))
    one_hot_labels = torch.eye(len(outputs[0]))[y].cuda()

    i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
    j, _ = torch.max((one_hot_labels) * outputs, dim=1)

    if targeted:
        return torch.clamp(i - j, min=0)

    else:
        return torch.clamp(j - i, min=0)


def get_probability(success_probability):
    probability = [v / sum(success_probability) for v in success_probability]
    return probability


def PPBA(model, x, y, index):
    def func(xs):
        z = torch.from_numpy(xs).float().cuda().view(-1, LOW_DIM)
        perturbation = (z @ Random_Matrix).view(z.shape[0], 3, image_size, image_size)
        new_image = (x + perturbation).clamp(0, 1)
        loss = cw_loss(new_image, y)
        loss = loss.cpu().numpy()
        return loss

    global N_query
    variables = LOW_DIM
    z = np.zeros((1, LOW_DIM))
    prev_f = func(z)[0]
    is_success = 0 if prev_f > 0 else 1
    effective_number = [
        np.ones((1, LOW_DIM)),
        np.ones((1, LOW_DIM)),
        np.ones((1, LOW_DIM)),
    ]
    ineffective_number = [
        np.ones((1, LOW_DIM)),
        np.ones((1, LOW_DIM)),
        np.ones((1, LOW_DIM)),
    ]

    for k in range(MAX_ITER):
        u = np.zeros((N_SAMPLES, LOW_DIM))
        r = np.random.uniform(size=(N_SAMPLES, LOW_DIM))
        effective_probability = [
            effective_number[i] / (effective_number[i] + ineffective_number[i])
            for i in range(len(effective_number))
        ]
        probability = get_probability(effective_probability)

        # u[r < probability[0] + probability[1]] = 0
        u[r < probability[0]] = -1
        u[r >= probability[0] + probability[1]] = 1

        uz = z + RHO * u

        uz_l2 = np.linalg.norm(uz, axis=1)
        uz = uz * np.minimum(1, 5 / uz_l2).reshape(-1, 1)

        fu = func(uz)
        if fu.min() < prev_f:
            worked_u = u[fu < prev_f]
            effective_probability[0] = effective_probability[0] * MOM + (
                worked_u == -1
            ).sum(0)
            effective_probability[1] = effective_probability[1] * MOM + (
                worked_u == 0
            ).sum(0)
            effective_probability[2] = effective_probability[2] * MOM + (
                worked_u == 1
            ).sum(0)
            not_worked_u = u[fu >= prev_f]
            ineffective_number[0] = ineffective_number[0] * MOM + (
                not_worked_u == -1
            ).sum(0)
            ineffective_number[1] = ineffective_number[1] * MOM + (
                not_worked_u == 0
            ).sum(0)
            ineffective_number[2] = ineffective_number[2] * MOM + (
                not_worked_u == 1
            ).sum(0)
            z = uz[np.argmin(fu)]
            prev_f = fu.min()
        else:
            ineffective_number[0] += (u == -1).sum(0)
            ineffective_number[1] += (u == 0).sum(0)
            ineffective_number[2] += (u == 1).sum(0)

        if prev_f <= 0:
            is_success = 1
            break

    current_q = N_query
    N_query = 0

    z = torch.from_numpy(z).float().cuda()
    perturbation = (z @ Random_Matrix).view(1, 3, image_size, image_size)
    new_image = (x + perturbation).clamp(0, 1)
    if current_q > MAX_ITER:
        is_success = 0
    print(index, current_q, is_success, perturbation.view(1, -1).norm(2, 1).item())
    return (
        current_q,
        is_success,
        perturbation.view(1, -1).norm(2, 1).item(),
        perturbation,
    )


model = getattr(models, MODEL)(pretrained=True).cuda()
model.eval()
if MODEL.startswith("inception"):
    image_size = 299
    testset = dset.ImageFolder(DATA_ROOT, utils.INCEPTION_TRANSFORM)
else:
    image_size = 224
    testset = dset.ImageFolder(DATA_ROOT, utils.IMAGENET_TRANSFORM)

if ORDER == "strided":
    Random_Matrix = np.zeros((LOW_DIM, 3 * image_size * image_size))
    indices = utils.block_order(image_size, 3, initial_size=FREQ_DIM, stride=STRIDE)
else:
    Random_Matrix = np.zeros((LOW_DIM, 3 * FREQ_DIM * FREQ_DIM))
    indices = random.sample(range(R), LOW_DIM)
for i in range(LOW_DIM):
    Random_Matrix[i][indices[i]] = 1

if ORDER == "strided":
    Random_Matrix = torch.from_numpy(Random_Matrix).view(-1, 3, image_size, image_size)
else:
    Random_Matrix = torch.from_numpy(Random_Matrix).view(-1, 3, FREQ_DIM, FREQ_DIM)


def expand_vector(x, size):
    batch_size = x.size(0)
    x = x.view(-1, 3, size, size)
    z = torch.zeros(batch_size, 3, image_size, image_size)
    z[:, :, :size, :size] = x
    return z


Random_Matrix = (
    utils.block_idct(
        expand_vector(
            Random_Matrix, size=(image_size if ORDER == "strided" else FREQ_DIM),
        ),
        block_size=image_size,
    )
    .view(LOW_DIM, -1)
    .cuda()
)


total_queries = []
total_success = []
NUM = args.num
for i in tqdm(range(NUM)):
    index = i
    image = testset[index][0].unsqueeze(0).cuda()
    label = model(normalize(image)).argmax(1)

    with torch.no_grad():
        q, s, l2, _ = PPBA(model, image, label, i)

    total_queries.append(q)
    total_success.append(s)
    print(i + 1, sum(total_queries) / (i + 1), sum(total_success) / (i + 1))
    print("=" * 10)


act_qs = [
    0 if total_success[i] == 0 or total_queries[i] > MAX_ITER else 1
    for i in range(len(total_success))
]
sum_qs = sum([act_qs[i] * total_queries[i] for i in range(len(total_queries))])
print("ASR:", sum(act_qs) / NUM)
print("On success:")
print("average queries:", sum_qs / sum(act_qs))

print("On all:")
print("average queries:", sum(total_queries) / NUM)
