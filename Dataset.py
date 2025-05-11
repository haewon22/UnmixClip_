import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

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
