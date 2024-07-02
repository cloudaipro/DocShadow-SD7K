import warnings
from accelerate import Accelerator
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
from config import Config
from models import *
from utils import *
import os
import cv2
import torch
from time import time

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)

path_img = "./input/2024-06-28_21_35_51-0 copy.jpg"
start = time()

img = cv2.imread(path_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # H, W, C

img_tensor = torch.tensor(img, dtype=torch.float32, device="cuda") # Convert to torch tensor
img_tensor = img_tensor / 255. # Normalize [0 - 1] range (but depends on the model)
img_tensor = img_tensor.permute(2, 0, 1) # Reorder to C, H, W (torch requires this format)
img_tensor = img_tensor.unsqueeze(0) # Becomes this format B, C, H, W

accelerator = Accelerator()

# inp = Image.open(path_img).convert('RGB')
# inp = np.array(inp)
# inp = F.to_tensor(img_tensor)

# Model & Metrics
model = Model()

print(opt.TESTING.WEIGHT)

load_checkpoint(model, opt.TESTING.WEIGHT)
model = model.to("cuda")
# model = accelerator.prepare(model)

model.eval()

if not os.path.exists("result"):
    os.makedirs("result")

with torch.no_grad(): # Dont run your gradients, speeds up inference

    traced_model = torch.jit.trace(model, img_tensor)
    # res = model(img_tensor)
    res = traced_model(img_tensor)

    save_image(res, os.path.join(os.getcwd(), "result", os.path.basename(path_img)))

print(f"Consumed time:{time() - start}")
