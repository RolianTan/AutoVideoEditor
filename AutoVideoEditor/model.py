import torch
import numpy as np
from torch import nn

class VisionBlock(nn.Module):
    def __init__(self, pretrain_model, input_dim, output_dim):
        super(VisionBlock, self).__init__()
        self.pretrain_model = pretrain_model
        self.fc = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, video_input):
        bs, v_len, c, h, w = video_input.shape
        # Reshape to (bs * v_len, c, h, w) for feature extraction
        video_input = video_input.view(bs * v_len, c, h, w)
        vision_features = self.pretrain_model(video_input).pooler_output
        # Reshape back to (bs, v_len, feature_dim)
        vision_features = vision_features.view(bs, v_len, -1)
        # Compute mean for each video's frames to aggregate features
        vision_features = vision_features.mean(dim=1)
        out = self.layer_norm(self.fc(vision_features))
        return out

class TextBlock(nn.Module):
    def __init__(self, pretrain_model, input_dim, output_dim):
        super(TextBlock, self).__init__()
        self.pretrain_model = pretrain_model
        self.fc = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, text_input):
        # Tokenize and encode text using the pre-trained CLIP text model
        text_features = self.pretrain_model(**text_input).pooler_output
        out = self.layer_norm(self.fc(text_features))
        return out

class AutoClipModel(nn.Module):
    def __init__(self,
                 vision_model,
                 vision_input_dim,
                 text_model,
                 text_input_dim,
                 projection_dim,
                 t=0.07):
        super(AutoClipModel, self).__init__()
        self.vision_model = VisionBlock(vision_model, vision_input_dim, projection_dim)
        self.text_model = TextBlock(text_model, text_input_dim, projection_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / t))

    def forward(self, video_input, text_input):
        vision_features = self.vision_model(video_input)
        text_features = self.text_model(text_input)

        # Normalize features
        vision_features = vision_features / vision_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity for contrastive loss
        logit_scale = self.logit_scale.exp()
        logit_video = logit_scale * vision_features @ text_features.t()
        logit_text = logit_video.t()

        return logit_video, logit_text