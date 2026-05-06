'''Model factory for creating different architectures.'''
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import CNN, ResNet18, VGG, Wide_ResNet


def create_model(model_name, model_args=None):
    """Create and return a model instance based on model name."""
    if model_args is None:
        model_args = {}
    
    num_classes = model_args.get('num_classes', 10)
    width_mult = model_args.get('width_mult', 1)
    
    # Normalize model name (handle lowercase aliases)
    model_name = model_name.lower() if model_name else model_name
    
    model_map = {
        'cnn': lambda: CNN(num_classes=num_classes),
        'resnet18': lambda: ResNet18(num_classes=num_classes),
        'vgg': lambda: VGG('VGG16', num_classes=num_classes),
        'vgg16': lambda: VGG('VGG16', num_classes=num_classes),
        'wideresnet': lambda: Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=num_classes),
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_map.keys())}")
    
    return model_map[model_name]()

def setup_model(device, model_name, model_args=None, use_data_parallel=True, cudnn_benchmark=True):
    """Create, move to device, and optionally wrap model with DataParallel."""
    if model_args is None:
        model_args = {}
    
    print('==> Building model..')
    
    # Create model
    net = create_model(model_name, model_args)
    
    # Move to device
    net = net.to(device)
    
    # Use DataParallel for multi-GPU
    if device == 'cuda' and use_data_parallel and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        print(f'Using {torch.cuda.device_count()} GPUs')
    
    # Enable cudnn benchmark
    if device == 'cuda' and cudnn_benchmark:
        cudnn.benchmark = True
    
    return net


def load_checkpoint(net, checkpoint_path):
    """Load model checkpoint."""
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(os.path.dirname(checkpoint_path)), f'Error: checkpoint directory not found!'
    
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
    
    print(f'Loaded checkpoint: Epoch {checkpoint["epoch"]}, Accuracy {checkpoint["acc"]:.2f}%')
    
    return best_acc, start_epoch


if __name__ == '__main__':
    """Test model factory."""
    print("Testing model factory...")
    
    # Test creating different models
    test_models = ['cnn', 'resnet18', 'vgg16', 'wideresnet']
    
    for model_name in test_models:
        try:
            print(f"\nCreating {model_name}...")
            net = create_model(model_name)
            print(f"  ✓ {model_name} created successfully!")
            print(f"  Parameters: {sum(p.numel() for p in net.parameters()):,}")
            
            # Test forward pass
            test_input = torch.randn(1, 3, 32, 32)
            output = net(test_input)
            print(f"  Output shape: {output.shape}")
        except Exception as e:
            print(f"  ✗ Error with {model_name}: {e}")
    
    print("\n✓ Model factory test completed!")


