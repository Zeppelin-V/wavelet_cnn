import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import numpy as np


def get_model(model_name, computing_device):

    if (model_name == "BaselineCNN"):
        model = BaselineCNN()
        model = model.to(computing_device)
        print("Model on CUDA?", next(model.paramaters()).is_cuda)
        return model
    elif (model_name == "WaveletCNN"):
        model= WaveletCNN()
        model = model.to(computing_device)
        print("Model on CUDA?", next(model.paramaters()).is_cuda)
        return model

    return 0


