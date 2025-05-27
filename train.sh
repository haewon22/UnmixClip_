python train.py \
  --data_root datasets/VOCdevkit/ \
  --train_split trainval \
  --val_split test \
  --out runs/unmix_clip \
  --batch 32 --log_int 1000 --epochs 50 --eval_freq 5 --run_name baseline

python train.py \
  --data_root datasets/VOCdevkit/ \
  --train_split trainval \
  --val_split test \
  --out runs/unmix_clip \
  --batch 32 --log_int 1000 --epochs 50 --alpha 0 --eval_freq 5 --run_name w/o_mfi

python train.py \
  --data_root datasets/VOCdevkit/ ㅅㄱ먀ㅜ\
  --train_split trainval \
  --val_split test \
  --out runs/unmix_clip \
  --batch 32 --log_int 1000 --epochs 50 --use_all_info 1 --eval_freq 5 --run_name use_all_info_ASL

python train.py \
  --data_root datasets/VOCdevkit/ \
  --train_split trainval \
  --val_split test \
  --out runs/unmix_clip \
  --batch 32 --log_int 1000 --epochs 50 --alpha 0 --use_all_info 1 --eval_freq 5 --run_name use_all_info_ASL_no_mfi