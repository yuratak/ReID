import os
import torch
import torch.nn as nn

def makedirs(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('path:%s created' %save_path)
    
    return


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    
    return


def save_checkpoint(model, save_path, device):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.to(device)

    return


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print("Invalid path!")

        return
    
    model.load_state_dict(torch.load(checkpoint_path))
    print('checkpoint_path:%s loaded' %checkpoint_path)
    model.to(device)

    return