import torch
import torch.nn as nn
import torch.nn.functional as F

class FuzzyBlock(nn.Module):
    def __init__(self, input_dim: int, fuzzy_dim: int):
        """
        Initializes the FuzzyBlock module.
        
        Arguments:
        input_dim (int): Dimension of the input features.
        fuzzy_dim (int): Dimension of the fuzzy feature space.
        """
        super(FuzzyBlock, self).__init__()
        # self.batch_norm = nn.BatchNorm1d(input_dim)
        self.gaussian_layer = nn.Linear(input_dim, input_dim)
        self.l2_norm = nn.LayerNorm(input_dim)
        self.fuzzy_layer = nn.Linear(input_dim, fuzzy_dim)
        self.mu_layer = nn.Linear(fuzzy_dim, fuzzy_dim)
        self.sigma_layer = nn.Linear(fuzzy_dim, fuzzy_dim)

    def gaussian_function(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian function"""
        return self.gaussian_layer(x)
    
    def gaussian_membership(self, x: torch.Tensor) -> torch.Tensor:
        """Computes fuzzy membership using Gaussian function"""
        mu = self.mu_layer(x)
        sigma = F.softplus(self.sigma_layer(x))
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def l2_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Applies L2 normalization"""
        return self.l2_norm(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FuzzyBlock module.
        
        Arguments:
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
        torch.Tensor: Output tensor with concatenated original and fuzzy features.
        """
        # x = self.batch_norm(x)
        x = self.gaussian_function(x)
        x = self.l2_normalization(x)
        fuzzy_features = self.fuzzy_layer(x)
        membership_x = self.gaussian_membership(fuzzy_features)
        return torch.cat([x, membership_x], dim=-1)

if __name__ == "__main__":
    # input_tensor = torch.randn(5, 10)
    # fuzzy_block = FuzzyBlock(input_dim=10, fuzzy_dim=5)
    dims = [96,192,384,768]
    for dim in dims:
        input_tensor = torch.randn([784, 64, dim])
        fuzzy_block = FuzzyBlock(input_dim=dim, fuzzy_dim=dim//2)
        output = fuzzy_block(input_tensor)
        output = nn.Linear(int(dim*1.5), 128)(output)
        output = nn.Dropout(.1)(output)
        output = nn.Linear(128, dim)(output)

        print("Output shape:", output.shape)
    # fuzzy_block = FuzzyBlock(input_dim=256, fuzzy_dim=128)
    # output = fuzzy_block(output)
    # print("Output shape:", output.shape)
    # print("Output:", output)