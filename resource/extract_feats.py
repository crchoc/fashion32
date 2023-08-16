from all_functions import *
from parameters import *
from tqdm import tqdm
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgba2rgb
from PIL import Image
import skimage.io
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import pickle as pkl
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()
