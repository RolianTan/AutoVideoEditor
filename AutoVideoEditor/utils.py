import torch
from torch import nn

import torch
import torch.nn as nn
import numpy as np
import cv2

# Contrastive loss function using precomputed similarity logits
def contrastive_loss(video_logits, text_logits, device, temperature=0.07):
    # Scale the logits by temperature
    video_logits = video_logits / temperature
    text_logits = text_logits / temperature

    # Labels: Identity matrix for correct matching pairs
    labels = torch.arange(video_logits.size(0)).long().to(device)

    # Cross-entropy loss function
    loss_func = nn.CrossEntropyLoss()

    # Compute loss in both directions: video -> text and text -> video
    loss = (loss_func(video_logits, labels) + loss_func(text_logits, labels)) / 2

    return loss

def load_dataloader(save_path):
    dataloader = torch.load(save_path)
    return dataloader



















