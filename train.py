import os, glob, json, argparse, logging, time
from typing import Tuple

import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode, RandAugment

import clip  # openai/CLIP

from model   import UnmixCLIP, MLPProjector, LinearProjector
from losses  import MFILoss, AsymmetricLoss
from Dataset import VOC2007Dataset         
from Cutout  import Cutout

import wandb

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")

# ─────────────────────────────────────────────────────────────
# utils
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False  # faster

# CLIP image‑preprocessing constants (OpenAI)
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD:  Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

# ─────────────────────────────────────────────────────────────
# mAP 계산
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def calc_map(pred: np.ndarray, target: np.ndarray) -> float:
    aps = []
    for k in range(target.shape[1]):
        if target[:, k].sum() == 0:
            continue  # class absent in batch/val set
        aps.append(average_precision_score(target[:, k], pred[:, k]))
    return float(np.mean(aps)) if aps else 0.0

# ─────────────────────────────────────────────────────────────
# 검증 루틴 (val_loss & mAP)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model: UnmixCLIP, loader: DataLoader, text_tokens: torch.Tensor,
             asl_loss_fn: AsymmetricLoss, mfi_loss_fn : MFILoss, device: torch.device):
    model.eval()
    tot_loss, asl_loss_val, mfi_loss_val, preds, gts = 0.0, 0.0, 0.0, [], [] # asl_loss, mfi_loss 변수명 변경

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # txt_proj_val 추가로 반환 받음
        lp, _,  txt_proj_val = model(imgs, text_tokens)
        asl = asl_loss_fn(lp, labels)

        mfi = mfi_loss_fn(txt_proj_val) # MFI Loss에 txt_proj_val 전달
        
        loss = asl + mfi
        tot_loss += loss.item()
        mfi_loss_val += mfi.item() # .item() 추가
        asl_loss_val += asl.item() # .item() 추가
        
        preds.append(torch.sigmoid(lp).cpu().detach().numpy())
        gts.append(labels.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts   = np.concatenate(gts,   axis=0)
    val_map = calc_map(preds, gts)

    return tot_loss / len(loader), asl_loss_val/len(loader), mfi_loss_val/len(loader), val_map # 반환값 변경

# ─────────────────────────────────────────────────────────────
# Dual Prompt 작성
# ─────────────────────────────────────────────────────────────

def build_dual_prompts(classes, custom_json=None):
    pos, neg = [], []
    for c in classes:
        if custom_json and c in custom_json:
            pos.append(custom_json[c]["positive"])
            neg.append(custom_json[c]["negative"])
        else:
            pos.append(f"There is a {c}.")
            neg.append(f"There is no {c}.")
    return pos + neg  # length = 2N

# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main(cfg):
    set_seed()

    # Device ---------------------------------------------------
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    # print("💻 device: %s"%device)
    logging.info(f"💻 device = {device}")

    # WandB ----------------------------------------------------
    wandb.init(project="unmix-clip", name=cfg.run_name, config=vars(cfg), resume="allow")

    # 1) CLIP RN‑101 backbone ---------------------------------
    clip_model, _ = clip.load("RN101", device=device, download_root=cfg.clip_root)
    if device == "mps":
        clip_model = clip_model.float()

    # freeze ALL CLIP parameters (visual + text)
    for p in clip_model.parameters():
        p.requires_grad = False

    # 2) Dataset & Dataloaders --------------------------------
    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size), interpolation=InterpolationMode.BICUBIC),
        RandAugment(num_ops=2, magnitude=9),
        Cutout(1, 32),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    voc_root = os.path.join(cfg.data_root, "VOC2007")
    train_set = VOC2007Dataset(voc_root, image_set=cfg.train_split, transform=train_tf)
    val_set   = VOC2007Dataset(voc_root, image_set=cfg.val_split,   transform=val_tf)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch, shuffle=True,
        num_workers=cfg.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch, shuffle=False,
        num_workers=cfg.workers, pin_memory=True,
    )

    # 3) Model -------------------------------------------------
    model = UnmixCLIP(
        clip_model,
        image_projector=LinearProjector(),
        text_projector=MLPProjector(),
    ).to(device)

    asl_loss = AsymmetricLoss().to(device)
    mfi_loss = MFILoss(lambda_=0.2).to(device)

    

    # only projectors are optimized
    optim_params = list(model.image_projector.parameters()) + list(model.text_projector.parameters())
    optimizer  = optim.SGD(optim_params, lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # 4) Prompts ----------------------------------------------
    custom_prompts = None
    if os.path.exists(cfg.prompts):
        with open(cfg.prompts, "r", encoding="utf-8") as f:
            custom_prompts = json.load(f)
    dual_prompts = build_dual_prompts(train_set.CLASSES, custom_prompts)
    text_tokens  = clip.tokenize(dual_prompts).to(device)

    # 5) Training loop ----------------------------------------
    os.makedirs(cfg.out, exist_ok=True)
    best_map      = 0.0
    global_step   = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, ncols=100, desc=f"Epoch {epoch}/{cfg.epochs}"):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lp, ln, txt_proj = model(imgs, text_tokens)

            # ASL Loss 계산 방식 수정:
            # lp는 긍정 로짓, ln은 부정 로짓으로 간주하고,
            # labels (긍정 타겟) 및 1-labels (부정 타겟)에 대해 각각 계산
            # 이 방식은 DualCoOp 논문의 Loss 구현 방식을 명확히 파악하여 적용하는 것이 가장 정확합니다.
            # 여기서는 일반적인 다중 레이블 학습의 관점에서 제안되었습니다.
            # train.py 학습 루프 내
        

            asl  = asl_loss(lp, labels)          # [B, N] × [B, N]
            mfi  = mfi_loss(txt_proj)
            loss = asl + cfg.alpha * mfi

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step  += 1

            if global_step % cfg.log_int == 0:
                wandb.log({"step_loss": loss.item(), "lr": scheduler.get_last_lr()[0], 'step_asl_loss': asl.item(), 'step_mfi_loss': mfi.item()}, step=global_step)

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch}: train_loss = {avg_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": avg_loss}, step=global_step)

        # ----- Validation ------------------------------------
        if epoch % cfg.eval_freq == 0 or epoch == cfg.epochs:
            val_loss, val_asl, val_mfi, val_map = validate(model, val_loader, text_tokens, asl_loss, mfi_loss, device)
            logging.info(f"  val_loss = {val_loss:.4f} | mAP = {val_map * 100:.2f}")
            wandb.log({"val_loss": val_loss, "val_asl" : val_asl, "val_mfi" : val_mfi,"mAP": val_map}, step=global_step)

            # checkpoint --------------------------------------
            ckpt = {
                "epoch": epoch,
                "step":  global_step,
                "model": {
                    "img": model.image_projector.state_dict(),
                    "txt": model.text_projector.state_dict(),
                },
                "optimizer": optimizer.state_dict(),
                "mAP": val_map,
            }
            name = f"epoch{epoch:02d}_map{val_map * 100:.1f}.pt"
            path = os.path.join(cfg.out, name)
            torch.save(ckpt, path)
            wandb.save(path)

            # best -------------------------------------------
            if val_map > best_map:
                best_map = val_map
                torch.save(ckpt, os.path.join(cfg.out, "best.pt"))

            # keep last 10 -----------------------------------
            ckpts = sorted(glob.glob(os.path.join(cfg.out, "epoch*.pt")), key=os.path.getmtime)
            for p in ckpts[:-10]:
                os.remove(p)

    logging.info(f"Training done.  Best mAP = {best_map * 100:.2f}")
    wandb.finish()

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def get_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="./VOCdataset/VOCdevkit/", help="Path to VOCdevkit directory (contains VOC2007)")
    p.add_argument("--train_split", default="trainval",   choices=["train", "val", "trainval", "test"], help="split name for training")
    p.add_argument("--val_split",   default="val",        choices=["train", "val", "trainval", "test"], help="split name for validation")
    p.add_argument("--prompts",     default="prompts.json", help="custom prompt json (optional)")
    p.add_argument("--clip_root",   default="clip_weights")
    p.add_argument("--out",         default="runs")
    p.add_argument("--run_name",    default="unmix‑clip")

    # training hyper‑params ----------------------------------
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--workers",  type=int,   default=15)
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--lr",       type=float, default=0.002)
    p.add_argument("--alpha",    type=float, default=7e-5, help="weight for MFI loss")
    p.add_argument("--img_size", type=int,   default=448,  help="square resize dimension")
    p.add_argument("--eval_freq",type=int,   default=5)
    p.add_argument("--log_int",  type=int,   default=100,  help="steps between WandB logs")
    p.add_argument("--use_all_info", type=int, default=0, help="use all information in the dataset")

    return p.parse_args()

if __name__ == "__main__":
    cfg = get_cfg()
    main(cfg)