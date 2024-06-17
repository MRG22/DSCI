import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]

        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            tags = " ".join(self.examples[i]['tags'])
            self.examples[i]['tags_index'] = tokenizer.encode(tags)

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        label = example['label']  # 加入label
        tags_id = example['tags_id']
        tags_index = example['tags_index']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, label, tags_id, tags_index)#label
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        label = example['label']  # 加入label
        tags_id = example['tags_id']
        tags_index = example['tags_index']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, label, tags_id, tags_index)
        return sample
