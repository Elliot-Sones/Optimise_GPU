"""
AlphaZero-style ResNet model for chess.

Architecture:
- Input: 18x8x8 board tensor
- Stem: 3x3 conv to 256 channels
- Body: 20 residual blocks
- Policy head: 1x1 conv → 4672 logits
- Value head: 1x1 conv → scalar in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from moves import NUM_ACTIONS


# ============================================================================
# Building blocks
# ============================================================================

class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual block with two conv layers.
    
    x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


# ============================================================================
# Policy and Value heads
# ============================================================================

class PolicyHead(nn.Module):
    """
    Policy head outputting 4672 action logits.
    
    x → Conv 1x1 (32 channels) → BN → ReLU → Flatten → Linear → 4672
    """
    
    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hidden_channels * 64, NUM_ACTIONS)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ValueHead(nn.Module):
    """
    Value head outputting scalar in [-1, 1].
    
    x → Conv 1x1 (1 channel) → BN → ReLU → Flatten → Linear(256) → ReLU → Linear(1) → Tanh
    """
    
    def __init__(self, in_channels: int, hidden_size: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x.squeeze(-1)


# ============================================================================
# Full model
# ============================================================================

class ChessResNet(nn.Module):
    """
    AlphaZero-style residual network for chess.
    
    Args:
        num_blocks: Number of residual blocks (default: 20)
        channels: Number of channels in residual blocks (default: 256)
        input_planes: Number of input planes (default: 18)
        policy_head_channels: Channels in policy head conv layer
        value_head_hidden: Hidden size in value head MLP
    """
    
    def __init__(
        self,
        num_blocks: int = 20,
        channels: int = 256,
        input_planes: int = 18,
        policy_head_channels: int = 32,
        value_head_hidden: int = 256,
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.channels = channels
        
        # Stem
        self.stem = ConvBNReLU(input_planes, channels, kernel_size=3, padding=1)
        
        # Residual tower
        self.blocks = nn.Sequential(*[
            ResidualBlock(channels) for _ in range(num_blocks)
        ])
        
        # Heads
        self.policy_head = PolicyHead(channels, policy_head_channels)
        self.value_head = ValueHead(channels, value_head_hidden)
    
    def forward(
        self, 
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Board tensor of shape (batch, 18, 8, 8)
            legal_mask: Optional legal move mask of shape (batch, 4672)
                        If provided, illegal moves get -inf logits
        
        Returns:
            policy_logits: Shape (batch, 4672)
            value: Shape (batch,)
        """
        # Shared trunk
        x = self.stem(x)
        x = self.blocks(x)
        
        # Heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        # Apply legal move mask
        if legal_mask is not None:
            # Set illegal moves to very negative value
            policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)
        
        return policy_logits, value
    
    def predict(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get policy probabilities and value.
        
        Args:
            x: Board tensor
            legal_mask: Legal move mask
            temperature: Softmax temperature (default 1.0)
        
        Returns:
            policy_probs: Shape (batch, 4672)
            value: Shape (batch,)
        """
        policy_logits, value = self.forward(x, legal_mask)
        policy_probs = F.softmax(policy_logits / temperature, dim=-1)
        return policy_probs, value
    
    def get_action(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor,
        greedy: bool = True,
    ) -> torch.Tensor:
        """
        Get action indices.
        
        Args:
            x: Board tensor of shape (batch, 18, 8, 8)
            legal_mask: Legal move mask
            greedy: If True, return argmax; else sample
        
        Returns:
            actions: Shape (batch,)
        """
        policy_logits, _ = self.forward(x, legal_mask)
        
        if greedy:
            return policy_logits.argmax(dim=-1)
        else:
            probs = F.softmax(policy_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ============================================================================
# Model variants
# ============================================================================

def create_model(
    variant: str = "medium",
    **kwargs,
) -> ChessResNet:
    """
    Create a model with preset configuration.
    
    Variants:
        tiny: 5 blocks, 64 channels (~200K params)
        small: 10 blocks, 128 channels (~2M params)
        medium: 20 blocks, 256 channels (~15M params)
        large: 40 blocks, 256 channels (~30M params)
    """
    configs = {
        "tiny": {"num_blocks": 5, "channels": 64},
        "small": {"num_blocks": 10, "channels": 128},
        "medium": {"num_blocks": 20, "channels": 256},
        "large": {"num_blocks": 40, "channels": 256},
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    
    config = {**configs[variant], **kwargs}
    return ChessResNet(**config)


# ============================================================================
# Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: ChessResNet):
    """Print model summary."""
    print(f"ChessResNet Configuration:")
    print(f"  Blocks: {model.num_blocks}")
    print(f"  Channels: {model.channels}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Flops/Step: {estimate_flops(model):.2e}")


def estimate_flops(model: ChessResNet) -> float:
    """
    Estimate max FLOPs per forward pass (batch size 1).
    
    Formula assumes:
    - Conv2d: 2 * Cin * Cout * K * K * H * W
    - Linear: 2 * Cin * Cout
    - BatchNorm/ReLU: Negligible
    """
    flops = 0
    H, W = 8, 8
    
    # Stem: Conv 3x3
    flops += 2 * 18 * model.channels * 3 * 3 * H * W
    
    # Residual blocks
    # Each block: 2 * Conv(C, C, 3, 3)
    block_flops = 2 * (2 * model.channels * model.channels * 3 * 3 * H * W)
    flops += model.num_blocks * block_flops
    
    # Policy Head
    # Conv 1x1: C -> 32
    flops += 2 * model.channels * 32 * 1 * 1 * H * W
    # Linear: 32*64 -> NUM_ACTIONS
    flops += 2 * (32 * 64) * NUM_ACTIONS
    
    # Value Head
    # Conv 1x1: C -> 1
    flops += 2 * model.channels * 1 * 1 * 1 * H * W
    # Linear: 64 -> 256
    flops += 2 * 64 * 256
    # Linear: 256 -> 1
    flops += 2 * 256 * 1
    
    return float(flops)
    
    # Test forward pass
    x = torch.randn(1, 18, 8, 8)
    policy, value = model(x)
    print(f"\nOutput shapes:")
    print(f"  Policy: {policy.shape}")
    print(f"  Value: {value.shape}")


# ============================================================================
# Testing
# ============================================================================

def test_model_forward():
    """Test model forward pass."""
    print("Testing model forward pass...")
    
    model = ChessResNet(num_blocks=5, channels=64)
    
    # Test batch
    batch_size = 8
    x = torch.randn(batch_size, 18, 8, 8)
    mask = torch.randint(0, 2, (batch_size, NUM_ACTIONS)).float()
    
    # Forward without mask
    policy, value = model(x)
    assert policy.shape == (batch_size, NUM_ACTIONS), f"Bad policy shape: {policy.shape}"
    assert value.shape == (batch_size,), f"Bad value shape: {value.shape}"
    
    # Forward with mask
    policy_masked, value2 = model(x, mask)
    assert policy_masked.shape == (batch_size, NUM_ACTIONS)
    
    # Check mask applied
    illegal_positions = (mask == 0)
    assert (policy_masked[illegal_positions] == -1e9).all(), "Mask not applied correctly"
    
    # Test predict
    probs, _ = model.predict(x, mask)
    assert probs.shape == (batch_size, NUM_ACTIONS)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    
    # Test get_action
    actions = model.get_action(x, mask, greedy=True)
    assert actions.shape == (batch_size,)
    
    print("✓ All forward pass tests passed!")
    return True


def test_model_gradients():
    """Test model can compute gradients."""
    print("Testing gradient computation...")
    
    model = ChessResNet(num_blocks=2, channels=32)
    
    x = torch.randn(4, 18, 8, 8)
    target_policy = torch.randint(0, NUM_ACTIONS, (4,))
    target_value = torch.randn(4)
    
    policy, value = model(x)
    
    # Compute loss
    policy_loss = F.cross_entropy(policy, target_policy)
    value_loss = F.mse_loss(value, target_value)
    total_loss = policy_loss + value_loss
    
    # Backward
    total_loss.backward()
    
    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
    
    print("✓ Gradient test passed!")
    return True


def test_model_variants():
    """Test all model variants."""
    print("Testing model variants...")
    
    for variant in ["tiny", "small", "medium"]:
        model = create_model(variant)
        x = torch.randn(2, 18, 8, 8)
        policy, value = model(x)
        params = count_parameters(model)
        print(f"  {variant}: {params:,} params")
    
    print("✓ All variants work!")
    return True


if __name__ == "__main__":
    test_model_forward()
    print()
    test_model_gradients()
    print()
    test_model_variants()
    print()
    
    # Print summary of default model
    print("=" * 50)
    model = create_model("medium")
    model_summary(model)
