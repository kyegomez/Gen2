
import torch
from gen2 import Gen2

model = Gen2()

images = torch.randn(1, 3, 128, 128)
video = torch.randn(1, 3, 16, 128, 128)

run_out = model.forward(images, video)
