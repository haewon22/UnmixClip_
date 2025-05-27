import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────
# 1) 텍스트용: MLPProjector (512 → 384 → 256)
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# 2) 이미지용: LinearProjector (512 → 256)
# ──────────────────────────────────────────────────────────────
# class LinearProjector(nn.Linear):
#     def __init__(self, input_dim=512, output_dim=256):
#         super().__init__(input_dim, output_dim, bias=False)
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
        B, P, D = x.shape            # [B, 49, 512]
        x = self.net(x.view(-1, D))
        return x.view(B, P, -1)
        



# ──────────────────────────────────────────────────────────────
# 3) Unmix-CLIP: 전체 모델
# ──────────────────────────────────────────────────────────────
class UnmixCLIP(nn.Module):
    def __init__(self, clip_model, image_projector, text_projector):
        super().__init__()
        self.clip = clip_model.visual              # ResNet-101 Encoder
        self.clip_text = clip_model.encode_text    # Text Encoder
        self.image_projector = image_projector     # Linear: 512→256
        self.text_projector = text_projector       # MLP:    512→384→256
        self.patch_proj_w = clip_model.visual.attnpool.c_proj.weight  # [512, 2048]
        self.patch_proj_w.requires_grad_(False)   

        for p in self.clip.parameters():
            p.requires_grad = False

    # ──────────────────────────────────────────────────────────
    # 3.1) CLIP 이미지 인코더에서 패치 벡터 (49개) 추출
    # ──────────────────────────────────────────────────────────
    def _forward_image_patches(self, images):
        """
        Input:  images [B, 3, 224, 224]
        Output: patch_embeds [B, 49, 512]
        """
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
        x = self.clip.layer4(x)           # [B, 2048, 14, 14]

        x = x.flatten(2).transpose(1, 2)  # (B, 196, 2048)

        x = x @ self.patch_proj_w.T        # or F.linear(x, self.patch_proj_w)

        return x

    # ──────────────────────────────────────────────────────────
    # 3.2) 모델 forward(images, text_tokens)
    # ──────────────────────────────────────────────────────────
    def forward(self, images, text_tokens):
        """
        images      : [B, 3, 224, 224]
        text_tokens : [2N, 77] (tokenized text prompts)
        Returns
        -------
        logits_pos, logits_neg : [B, N] × 2
        txt_proj               : [2N, 256] (for MFI loss)
        """
        # ① 이미지 인코딩
        with torch.no_grad():
            patches = self._forward_image_patches(images)  # [B, 49, 512]
        img_proj = self.image_projector(patches)           # [B, 49, 256]
        img_proj = F.normalize(img_proj, dim=-1)

        # ② 텍스트 인코딩 (고정)
        with torch.no_grad():
            txt_feat = self.clip_text(text_tokens)         # [2N, 512]
        txt_proj = self.text_projector(txt_feat)           # [2N, 256]
        txt_proj = F.normalize(txt_proj, dim=-1)

        # ③ Conv1d로 cosine 유사도 계산: [B, 2N, 49]
        O = 20 * F.conv1d(
            img_proj.transpose(1, 2),      # [B, 256, 49]
            txt_proj[:, :, None]           # [2N, 256, 1]
        )                                  # [B, 2N, 49]

        # ④ soft-weighted sum (DualCoOp 방식)
        W = F.softmax(O, dim=-1)           # [B, 2N, 49]
        logits = 5* (W * O).sum(-1)       # [B, 2N]  # 5 * 

        # ⑤ 양/음 분리 (dual prompt)
        B, _2N = logits.size()
        N = _2N // 2
        logits = logits.view(B, 2, N)      # [B, 2, N]
        logits_pos = logits[:, 0]          # [B, N]
        logits_neg = logits[:, 1]          # [B, N]

        return logits_pos, logits_neg, txt_proj