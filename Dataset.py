import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

import xml.etree.ElementTree as ET

class VOC2007Dataset(Dataset):

    CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]
    NAME2IDX = {name: idx for idx, name in enumerate(CLASSES)}

    def __init__(self, root, image_set="trainval", transform=None):
        self.root = root
        self.transform = transform

        split_file = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")
        if not os.path.isfile(split_file):
            raise RuntimeError(f"Split file not found: {split_file}")

        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.num_classes = len(self.CLASSES)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        img_path = os.path.join(self.root, "JPEGImages", f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        anno_path = os.path.join(self.root, "Annotations", f"{img_id}.xml")
        target_vec = torch.zeros(self.num_classes)

        tree = ET.parse(anno_path)
        for obj in tree.findall("object"):
            cls_name = obj.find("name").text.lower().strip()
            if cls_name in self.NAME2IDX:
                target_vec[self.NAME2IDX[cls_name]] = 1

        return image, target_vec


class Coco14Dataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.coco = COCO(annFile)
        self.root = root
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])

        self.cat2idx = {cat['id']: idx for idx, cat in enumerate(cats)}
        self.num_classes = len(self.cat2idx)

        self.classes = [cat['name'] for cat in cats]
        
    def __len__(self):
        return len(self.ids)  # 80
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        target = torch.zeros(self.num_classes)
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in self.cat2idx:
                idx = self.cat2idx[cat_id]
                target[idx] = 1

        return image, target
        