import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.model import ConvTran  

class BlurLayer(nn.Module):
    ''' 
    A layer that applies a fixed Gaussian-like blur using a predefined kernel.
    '''
    def __init__(self):
        super(BlurLayer, self).__init__()
        kernel = torch.tensor([[1.0, 2.0, 1.0],
                               [2.0, 4.0, 2.0],
                               [1.0, 2.0, 1.0]], dtype=torch.float32)
        kernel /= kernel.sum()
        self.register_buffer('blur_kernel', kernel.view(1, 1, 3, 3))

    def forward(self, x):
        '''
        Apply blur to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Blurred tensor.
        '''
        b, c, h, w = x.shape
        kernel = self.blur_kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=c)

class ModifiedVGG8(nn.Module):
    '''
    Modified VGG8 architecture using GELU activation and BlurLayer.
    '''
    def __init__(self):
        super(ModifiedVGG8, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.GELU(),
            BlurLayer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            BlurLayer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            BlurLayer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.GELU(),
            BlurLayer(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        '''
        Forward pass through ModifiedVGG8.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Feature tensor.
        '''
        return self.features(x)

class StandardVGG8(nn.Module):
    '''
    Standard VGG8 architecture using ReLU activation without blur.
    '''
    def __init__(self):
        super(StandardVGG8, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        '''
        Forward pass through StandardVGG8.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Feature tensor.
        '''
        return self.features(x)

class CustomModel(nn.Module):
    '''
    Custom model that integrates image features and numerical inputs for regression.
    
    Args:
        img_size (tuple): Image dimensions.
        sliding_window (int): Number of frames per sample.
        max_label (float): Maximum label value.
        min_label (float): Minimum label value.
        config (dict): Configuration for ConvTran.
        vgg_type (str): Type of VGG8 to use ("modified" or "standard").
    '''
    def __init__(self, img_size, sliding_window, max_label, min_label, config, vgg_type="modified"):
        super(CustomModel, self).__init__()
        if vgg_type == "modified":
            self.vgg8 = ModifiedVGG8()
        else:
            self.vgg8 = StandardVGG8()
        # Determine feature dimension using a dummy input.
        dummy_input = torch.zeros(1, 1, img_size[1], img_size[0])
        dummy_output = self.vgg8(dummy_input)
        _, C, H, W = dummy_output.size()
        # Process numerical input.
        self.number_dense = nn.Linear(1, 512)
        # Initialize ConvTran.
        self.convtran = ConvTran(config, num_classes=1)
        self.max_label = max_label
        self.min_label = min_label

    def forward(self, image_input, number_input):
        '''
        Forward pass combining image features and numerical inputs.
        
        Args:
            image_input (torch.Tensor): Tensor with shape (batch, sliding_window, 1, H, W).
            number_input (torch.Tensor): Tensor with shape (batch, sliding_window).
            
        Returns:
            torch.Tensor: Model predictions.
        '''
        batch_size, timesteps, C, H, W = image_input.size()
        image_input = image_input.view(batch_size * timesteps, C, H, W)
        x = self.vgg8(image_input)
        x = x.mean(dim=[-2, -1])
        x = x.view(batch_size, timesteps, -1)
        number_input = number_input.unsqueeze(-1)
        number_features = self.number_dense(number_input)
        x = x + number_features
        x = x.permute(0, 2, 1)
        x = self.convtran(x)
        outputs = ((x + 1) / 2) * (self.max_label - self.min_label) + self.min_label
        outputs = torch.mean(outputs, dim=1, keepdim=True)
        return outputs

