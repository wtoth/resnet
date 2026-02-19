import torch 
from torch import nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 64 channel residual layers 
        self.res64_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res64_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res64_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

       # 128 channel residual layers 
        self.res128_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res128_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res128_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res128_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        # 256 channel residual layers 
        self.res256_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res256_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res256_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res256_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res256_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res256_layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        # 512 channel residual layers 
        self.res512_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res512_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.res512_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        self.avg_pool = nn.AvgPool2d(2)

        self.output = nn.Linear(in_features=3*3*512, out_features=1000) 

        self.init_weights()


    def forward(self, x):
        # input layer
        input_layer_output = self.input_layer(x)

        # 64 Channel Layers 
        res64_layer1_output = self.res64_layer1(input_layer_output) # convolution - F(x)
        #print(input_layer_output.size(), res64_layer1_output.size())
        res64_layer1_residuals = F.relu(res64_layer1_output + input_layer_output) # residual - F(x) + x

        res64_layer2_output = self.res64_layer2(res64_layer1_residuals) # convolution - F(x)
        res64_layer2_residuals = F.relu(res64_layer2_output + res64_layer1_residuals) # residual - F(x) + x

        res64_layer3_output = self.res64_layer3(res64_layer2_residuals) # convolution - F(x)
        res64_layer3_residuals = F.relu(res64_layer3_output + res64_layer2_residuals) # residual - F(x) + x

        # 128 Channels Layers
        res128_layer1_output = self.res128_layer1(res64_layer3_residuals) # convolution - F(x)
        #transform residuals from 64 channels to 128 
        reshaped_64_residuals = res64_layer3_residuals.reshape(res64_layer3_residuals.shape[0], res64_layer3_residuals.shape[1]*2,res64_layer3_residuals.shape[2]//2,-1)
        reshaped_64_residuals = reshaped_64_residuals[:,:,:,:reshaped_64_residuals.shape[2]]
        res128_layer1_residuals = F.relu(res128_layer1_output + reshaped_64_residuals) # residual - F(x) + x

        res128_layer2_output = self.res128_layer2(res128_layer1_residuals) # convolution - F(x)
        res128_layer2_residuals = F.relu(res128_layer2_output + res128_layer1_residuals) # residual - F(x) + x

        res128_layer3_output = self.res128_layer3(res128_layer2_residuals) # convolution - F(x)
        res128_layer3_residuals = F.relu(res128_layer3_output + res128_layer2_residuals) # residual - F(x) + x

        res128_layer4_output = self.res128_layer4(res128_layer3_residuals) # convolution - F(x)
        res128_layer4_residuals = F.relu(res128_layer4_output + res128_layer3_residuals) # residual - F(x) + x

        # 256 Channels Layers
        res256_layer1_output = self.res256_layer1(res128_layer4_residuals) # convolution - F(x)
        #transform residuals from 64 channels to 128 
        reshaped_128_residuals = res128_layer4_residuals.reshape(res128_layer4_residuals.shape[0], res128_layer4_residuals.shape[1]*2,res128_layer4_residuals.shape[2]//2,-1)
        reshaped_128_residuals = reshaped_128_residuals[:,:,:,:reshaped_128_residuals.shape[2]]
        res256_layer1_residuals = F.relu(res256_layer1_output + reshaped_128_residuals) # residual - F(x) + x

        res256_layer2_output = self.res256_layer2(res256_layer1_residuals) # convolution - F(x)
        res256_layer2_residuals = F.relu(res256_layer2_output + res256_layer1_residuals) # residual - F(x) + x

        res256_layer3_output = self.res256_layer3(res256_layer2_residuals) # convolution - F(x)
        res256_layer3_residuals = F.relu(res256_layer3_output + res256_layer2_residuals) # residual - F(x) + x

        res256_layer4_output = self.res256_layer4(res256_layer3_residuals) # convolution - F(x)
        res256_layer4_residuals = F.relu(res256_layer4_output + res256_layer3_residuals) # residual - F(x) + x

        res256_layer5_output = self.res256_layer4(res256_layer4_residuals) # convolution - F(x)
        res256_layer5_residuals = F.relu(res256_layer5_output + res256_layer4_residuals) # residual - F(x) + x

        res256_layer6_output = self.res256_layer6(res256_layer5_residuals) # convolution - F(x)
        res256_layer6_residuals = F.relu(res256_layer6_output + res256_layer5_residuals) # residual - F(x) + x

        # 512 Channels Layers
        res512_layer1_output = self.res512_layer1(res256_layer6_residuals) # convolution - F(x)
        #transform residuals from 64 channels to 128 
        reshaped_256_residuals = res256_layer6_residuals.reshape(res256_layer6_residuals.shape[0], res256_layer6_residuals.shape[1]*2,res256_layer6_residuals.shape[2]//2,-1)
        reshaped_256_residuals = reshaped_256_residuals[:,:,:,:reshaped_256_residuals.shape[2]]
        res512_layer1_residuals = F.relu(res512_layer1_output + reshaped_256_residuals) # residual - F(x) + x

        res512_layer2_output = self.res512_layer2(res512_layer1_residuals) # convolution - F(x)
        res512_layer2_residuals = F.relu(res512_layer2_output + res512_layer1_residuals) # residual - F(x) + x

        res512_layer3_output = self.res512_layer3(res512_layer2_residuals) # convolution - F(x)
        res512_layer3_residuals = F.relu(res512_layer3_output + res512_layer2_residuals) # residual - F(x) + x

        # avg pool and output layer 
        avg_pool = self.avg_pool(res512_layer3_residuals)
        flattened_avg_pool = avg_pool.flatten(1) # flatten to a single dimension for the linear matmul
        output = self.output(flattened_avg_pool)

        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)