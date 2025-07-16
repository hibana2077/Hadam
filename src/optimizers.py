"""
Optimizer creation utilities.
"""

import torch


def create_optimizer(model: torch.nn.Module, config: dict):
    """Create optimizer based on configuration."""
    opt_name = config['optimizer'].lower()
    lr = config['lr']
    
    if opt_name == 'hadam':
        try:
            from opt.hadam import HAdam
            optimizer = HAdam(
                model.parameters(),
                lr=lr,
                order=config['hadam']['order']
            )
        except ImportError:
            print("Warning: Hadam optimizer not found, using SGD instead")
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=config['sgd']['momentum'],
                weight_decay=config['sgd']['weight_decay']
            )
    
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config['sgd']['momentum'],
            weight_decay=config['sgd']['weight_decay']
        )
    
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=config['adam']['betas'],
            eps=config['adam']['eps'],
            weight_decay=config['adam']['weight_decay']
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    return optimizer
