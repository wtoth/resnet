# ResNets (Deep Residual Learning for Image Recognition)
This is my replication of the ResNet Model using the ImageNet Dataset


# Model Specification 
Due to GPU Size constraints I chose to implement ResNet-34 in both the Resnet-34 A and ResNet-34 B options

## Setup and Running
This project was created using `uv` and is highly recommended<br>
After installing `uv` this project should run out of the box<br>

### Data Setup
First, you should download the ImageNet Dataset from Kaggle [here](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)<br>
Once this is downloaded you will need to run the `create_dataset_csv()` script to create our datasets<br>
You will need to set the `data` variable to your ImageNet location
```
uv run create_dataset_csv.py train
uv run create_dataset_csv.py val
```

### Training
Before kicking off training you should update the weights and biases variables `entity` and `project` in `init_logging()` in train.py to match your account.<br>
If not using Weights and Biases (not recommended) you can set logs to `False` in main.py<br>
To kick off training you can run<br>
```uv run main.py```

# Citation 
Paper Arxiv Link ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) <br>
CVPR [Link](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

```bibtex
@InProceedings{He_2016_CVPR,
author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
title = {Deep Residual Learning for Image Recognition},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2016}
}
```
