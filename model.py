class UnmixCLIP(nn.Module):
    ...
    # ──────────────────────────────────────────────────────
    # 3.1) 패치 추출 (입력 해상도에 따라 자동으로 7×7 또는 14×14)
    # ──────────────────────────────────────────────────────
    def _forward_image_patches(self, images):
        """
        images:  [B, 3, H, W]  (224 또는 448 권장)
        224×224 → attnpool 출력   : CLS + 7×7 = 50 토큰
        448×448 → attnpool 출력   : CLS + 14×14 = 197 토큰
        return  : patch_tokens   : [B, HW, 512]
                                   49  (224)  또는 196 (448)
        """
        x = self.clip.conv1(images)
        x = self.clip.bn1(x);  x = self.clip.relu1(x)
        x = self.clip.conv2(x); x = self.clip.bn2(x); x = self.clip.relu2(x)
        x = self.clip.conv3(x); x = self.clip.bn3(x); x = self.clip.relu3(x)

        x = self.clip.avgpool(x)
        x = self.clip.layer1(x); x = self.clip.layer2(x)
        x = self.clip.layer3(x); x = self.clip.layer4(x)   # [B, 2048, H/32, W/32]

        x = self.clip.attnpool(x)          # [B, 1+HW, 512]
        patch_tokens = x[:, 1:, :]         # CLS 제외 ⇒ [B, HW, 512]
        # 예) 224px → (B, 49, 512) , 448px → (B, 196, 512)
        # 448×448 / 32 = 14×14
        return patch_tokens
    ...
    def forward(self, images, text_tokens):
        """
        images      : [B, 3, H, W]  (H=W=224 또는 448)
        text_tokens : [2N, 77]
        returns
        --------
        logits_pos  : [B, N]
        logits_neg  : [B, N]
        txt_proj    : [2N, 256] (MFI 용)
        """
        # ① 이미지 인코딩
        with torch.no_grad():
            patches = self._forward_image_patches(images)   # [B, HW, 512]
        img_proj = self.image_projector(patches)            # [B, HW, 256]
        img_proj = F.normalize(img_proj, dim=-1)

        # ② 텍스트 인코딩
        with torch.no_grad():
            txt_feat = self.clip_text(text_tokens)          # [2N, 512]
        txt_proj = self.text_projector(txt_feat)            # [2N, 256]
        txt_proj = F.normalize(txt_proj, dim=-1)

        # ③ 패치×텍스트 유사도 : conv1d
        # img_proj^T : [B, 256, HW]  ──> HW = 49 또는 196
        O = 20 * F.conv1d(img_proj.transpose(1, 2),
                          txt_proj[:, :, None])             # [B, 2N, HW]

        W = F.softmax(O, dim=-1)                            # [B, 2N, HW]
        logits = 5 * (O * W).sum(-1)                        # [B, 2N]

        # ④ dual prompt 분리
        B, _2N = logits.shape
        N = _2N // 2
        logits = logits.view(B, 2, N)                       # [B, 2, N]
        return logits[:, 0], logits[:, 1], txt_proj          # pos, neg, txt_proj