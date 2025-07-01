# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLPProjector(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=(384,), output_dim=256):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < len(dims) - 2:
                layers.extend([nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LinearProjector(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=384, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, x):
        B, P, D = x.shape
        x = self.net(x.view(-1, D))
        return x.view(B, P, -1)

class UnmixCLIP(nn.Module):
    def __init__(self, clip_model, image_projector, text_projector):
        super().__init__()
        self.clip = clip_model.visual
        self.clip_text = clip_model.encode_text
        self.image_projector = image_projector
        self.text_projector = text_projector
        self.patch_proj_w = clip_model.visual.attnpool.c_proj.weight
        self.patch_proj_w_bias = clip_model.visual.attnpool.c_proj.bias
        self.patch_proj_w.requires_grad_(False)
        self.patch_proj_v = clip_model.visual.attnpool.v_proj.weight
        self.patch_proj_v_bias = clip_model.visual.attnpool.v_proj.bias
        self.patch_proj_v.requires_grad_(False)
        for p in self.clip.parameters():
            p.requires_grad = False

    def _forward_image_patches(self, images):
        x = self.clip.conv1(images)
        x = self.clip.bn1(x)
        x = self.clip.relu1(x)
        x = self.clip.conv2(x)
        x = self.clip.bn2(x)
        x = self.clip.relu2(x)
        x = self.clip.conv3(x)
        x = self.clip.bn3(x)
        x = self.clip.relu3(x)
        x = self.clip.avgpool(x)
        x = self.clip.layer1(x)
        x = self.clip.layer2(x)
        x = self.clip.layer3(x)
        x = self.clip.layer4(x)

        x = x.flatten(2).transpose(1, 2)     
        x = x @ self.patch_proj_v.T          
        x = x + self.patch_proj_v_bias       
        return x                             

    def forward(self, images, text_tokens):
        with torch.no_grad():
            patches = self._forward_image_patches(images)
        img_proj = self.image_projector(patches)
        img_proj = F.normalize(img_proj, dim=-1)

        with torch.no_grad():
            txt_feat = self.clip_text(text_tokens)
        txt_proj = self.text_projector(txt_feat)
        txt_proj = F.normalize(txt_proj, dim=-1)

        O = 20 * F.conv1d(img_proj.transpose(1, 2),
                         txt_proj[:, :, None])
        W = F.softmax(O, dim=-1)
        logits = 3 * (W * O).sum(-1)
        B, _2N = logits.size()
        N = _2N // 2
        logits = logits.view(B, 2, N)
        return logits[:,0], logits[:,1], txt_proj

    def encode_image(self, images):
        with torch.no_grad():
            patches = self._forward_image_patches(images)
        img_proj = self.image_projector(patches)
        return F.normalize(img_proj, dim=-1)

    def encode_text(self, text_tokens):
        with torch.no_grad():
            txt_feat = self.clip_text(text_tokens)
        txt_proj = self.text_projector(txt_feat)
        return F.normalize(txt_proj, dim=-1)
