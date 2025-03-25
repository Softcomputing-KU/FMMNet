import torch
import torch.nn as nn
import torch.nn.functional as F

class PossibilityMinMaxLayer(nn.Module):
    def __init__(self, in_channels, out_channels, gamma=1.0):
        super(PossibilityMinMaxLayer, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))  # Learnable gamma
        self.conv_V = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_W = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Learnable edge detection filters
        self.edge_detector_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.edge_detector_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

        # Initialization with Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # Apply Sobel filters to each channel
        self.edge_detector_x.weight.data = sobel_x.repeat(in_channels, 1, 1, 1)
        self.edge_detector_y.weight.data = sobel_y.repeat(in_channels, 1, 1, 1)

        self.learnable_gradient_scale = nn.Parameter(torch.tensor(0.5))  # Learnable scale
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        V = self.conv_V(x)
        W = self.conv_W(x)

        # Gradient calculation using learnable edge detectors
        grad_x = self.edge_detector_x(x)
        grad_y = self.edge_detector_y(x)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalization and applying learnable weights
        gradient_magnitude = self.norm(gradient_magnitude)
        gradient_weight = 1 + self.learnable_gradient_scale * gradient_magnitude

        delta = gradient_weight * torch.max(F.relu(self.gamma * (V - x)),
                                            F.relu(self.gamma * (x - W)))

        membership = 1 - delta  # Delta integrated with edge information
        return membership

if __name__ == "__main__":
    # input_tensor = torch.randn(5, 10)
    # fuzzy_block = FuzzyBlock(input_dim=10, fuzzy_dim=5)
    input_tensor = torch.randn([128, 192, 28, 28])
    PMM = PossibilityMinMaxLayer(192, 192)
    output = PMM(input_tensor)
    # output = nn.Linear(int(dim*1.5), 128)(output)
    # output = nn.Dropout(.1)(output)
    # output = nn.Linear(128, dim)(output)

    print("Output shape:", output.shape)
    # fuzzy_block = FuzzyBlock(input_dim=256, fuzzy_dim=128)
    # output = fuzzy_block(output)
    # print("Output shape:", output.shape)
    # print("Output:", output)