import torch
from train import ResNetModel
from torchvision.transforms import v2
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def main():
    root_directory = "/Volumes/PrizeNVMe/imagenet/ILSVRC/"

    if torch.backends.mps.is_available():
        print("mps")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("cuda")
        device = torch.device("cuda")
    else:
        print("cpu")
        device = torch.device("cpu")

    # Hyperparams
    num_epochs = 40
    batch_size = 64 # the paper used 256 but they probably had a bigger gpu
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 1e-4

    spatial_transforms = v2.Compose([  
        v2.Resize(size=(480,480)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(size=(224,224)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float, scale=False),
        v2.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

    color_transforms = v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    validation_transforms = v2.Compose([
        v2.PILToTensor(),
    ])

    resnet_model = ResNetModel(device, log=False)
    resnet_model.train(root_directory, num_epochs, batch_size, learning_rate, momentum, weight_decay, spatial_transforms, color_transforms, validation_transforms)

if __name__ == "__main__":
    main()