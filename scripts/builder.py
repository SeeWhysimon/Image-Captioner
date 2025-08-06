import torch
import torchvision.transforms as transforms

def build_transform(cfg):
    transform_list = []

    # Resize
    if 'resize' in cfg:
        transform_list.append(transforms.Resize(cfg['resize']))

    # ToTensor
    if cfg.get('to_tensor', True):
        transform_list.append(transforms.ToTensor())

    # Normalize
    if 'normalize' in cfg:
        norm = cfg['normalize']
        transform_list.append(transforms.Normalize(mean=norm['mean'], std=norm['std']))

    return transforms.Compose(transform_list)


def build_optimizer(cfg, model):
    optim_cfg = cfg.get("optimizer", {})

    # optimizer type (default is Adam)
    optimizer_type = optim_cfg.get("type", "Adam")

    # learning rate (default is 1e-3)
    lr = optim_cfg.get("lr", 1e-3)

    # weight decay (default is 0)
    weight_decay = optim_cfg.get("weight_decay", 0)

    if optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def build_scheduler(cfg, optimizer):
    sched_cfg = cfg.get("scheduler", {})
    
    # scheduler type (default is StepLR)
    scheduler_type = sched_cfg.get("type", "StepLR")
    
    # step size (default is 10)
    step_size = sched_cfg.get("step_size", 10)
    
    # gamma (default is 0.1)
    gamma = sched_cfg.get("gamma", 0.1)

    if scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")