import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import random
from scipy.ndimage import rotate
import imageio
from skimage.transform import resize
from collections import defaultdict
import csv

class MultiModalImbalancedDataset(Dataset):
    def __init__(self, split_id):

        self.classes = [0,1,2]
        self.samples = defaultdict(lambda: defaultdict(list))
        self.label_path='/memory/wangqimei/polyp_data/Piccolo/clinical metadata_release0.1.csv'
        
        # self.label_path=''
        self.label_dict={}
        with open(self.label_path, 'r',encoding='utf-8') as file:
            reader = csv.reader(file)

            for row in reader:
                idx=row[0].split(';')[0]
                disease=row[0].split(';')[7]
                self.label_dict[idx]=disease
        char_to_num = {
            'Hyperplasia': 0,
            'Adenoma': 1,
            'Adenocarcinoma': 2
        }
        # self.white_list_all = open('').readlines()
        # self.nbi_list_all = open('').readlines()
        # self.white_list_pair= open('').readlines()
        # self.nbi_list_pair = open('').readlines()
        self.white_list_all = open('/memory/wangqimei/train_data_path/Piccolo_all/WL/%d/Trainset/images.txt' % split_id).readlines()
        self.nbi_list_all = open('/memory/wangqimei/train_data_path/Piccolo_all/NBI/%d/Trainset/images.txt' % split_id).readlines()
        self.white_list_pair= open('/memory/wangqimei/train_data_path/Piccolo_pair_5fold/WL/%d/Trainset/images.txt' % split_id).readlines()
        self.nbi_list_pair = open('/memory/wangqimei/train_data_path/Piccolo_pair_5fold/NBI/%d/Trainset/images.txt' % split_id).readlines()
        self.white_list_pair=[x.replace('Piccolo/','Piccolo/polyps_crop_split_modality_all/').replace('Trainset','train/images').replace('Testset','test/images').replace('Valset','val/images') for x in self.white_list_pair]
        self.white_list= list(set(self.white_list_all) - set(self.white_list_pair))

        self.nbi_list= self.nbi_list_all

        self.white_list = list(map(lambda x: x.strip(), self.white_list))
        self.nbi_list = list(map(lambda x: x.strip(), self.nbi_list))
        random.shuffle(self.white_list)
        random.shuffle(self.nbi_list)
        for modality in ['WL','NBI']:
            if modality=='WL':
                img_list=self.white_list
            else:
                img_list=self.nbi_list
            for img_path in img_list:
                cls=char_to_num[self.label_dict[img_path.split('/')[-1].split('_')[0].lstrip('0')]]
                self.samples[modality][cls].append(img_path)
        
        self.class_counts = {}
        self.modalities=['WL','NBI']
        for cls in self.classes:
            min_count = min(len(self.samples[m][cls]) for m in self.modalities)
            self.class_counts[cls] = min_count

        self.pair_indices = []
        self.reset_pair_indices()
    
    def reset_pair_indices(self):
        self.pair_indices = []
        
        for cls in self.classes:
            for m in self.modalities:
                random.shuffle(self.samples[m][cls])
            for i in range(self.class_counts[cls]):
                pair = {
                    'class': cls,
                    'modalities': {}
                }
                
                for modality in self.modalities:
                    path = self.samples[modality][cls][i % len(self.samples[modality][cls])]
                    pair['modalities'][modality] = path
                self.pair_indices.append(pair)
    
        random.shuffle(self.pair_indices)

    def __len__(self):
        return len(self.pair_indices)
    
    def __getitem__(self, idx):
        pair = self.pair_indices[idx]
        data = {'class': pair['class']}
        
        for modality in self.modalities:
            img_path = pair['modalities'][modality]
            image= imageio.imread(img_path)

            angle = random.randint(-180, 180)
            image = rotate(image, angle)


            if random.random() > 0.5:
                image = np.flip(image, 0)

            if random.random() > 0.5:
                image = np.flip(image, 1)
            # resize
            image = resize(image, (448, 448))
            # swap axis
            image = np.swapaxes(image, 0,-1)
            
            data[modality] = torch.tensor(image)
        return data
    
    def get_balanced_sampler(self):
        class_weights = {cls: 1.0 / count for cls, count in self.class_counts.items()}
        sample_weights = [class_weights[pair['class']] for pair in self.pair_indices]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.pair_indices),  
            replacement=True  
        )
        return sampler


class EpochAwareDataLoader:

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.dataloader_args = kwargs
        self.dataloader = None
        self.reset_dataloader()
    
    def reset_dataloader(self):

        self.dataset.reset_pair_indices()
        sampler = self.dataset.get_balanced_sampler()
        self.dataloader = DataLoader(
            self.dataset,
            sampler=sampler,
            **self.dataloader_args
        )
    
    def __iter__(self):
        self.reset_dataloader()
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
def collate_fn(batch):
    collated = {'class': torch.tensor([item['class'] for item in batch])}
    modalities = set()
    for item in batch:
        modalities.update(item.keys())
    modalities.discard('class')
    
    for modality in modalities:
        collated[modality] = torch.stack([item[modality] for item in batch])
    
    return collated


class PolyDataset_nocrop_5fold_pair(Dataset):
    def __init__(self, is_train, split_id,enable_aug=True):
        self.enable_aug = enable_aug
        self.is_train=is_train
        self.label_path='/memory/wangqimei/polyp_data/Piccolo/clinical metadata_release0.1.csv'
        self.label_dict={}
        with open(self.label_path, 'r',encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                idx=row[0].split(';')[0]
                disease=row[0].split(';')[7]
                self.label_dict[idx]=disease

        if is_train:
            self.white_list= open('/memory/wangqimei/train_data_path/Piccolo_pair_5fold/WL/%d/Trainset/images.txt' % split_id).readlines()
            self.nbi_list = open('/memory/wangqimei/train_data_path/Piccolo_pair_5fold/NBI/%d/Trainset/images.txt' % split_id).readlines()
            # self.white_list = open('/memory/wangqimei/train_data_path/private_polyps_ad_hp_WL_NBI_with_mask/WL/TrainDataset/images.txt').readlines()
            # self.nbi_list = open('/memory/wangqimei/train_data_path/private_polyps_ad_hp_WL_NBI_with_mask/NBI/TrainDataset/images.txt').readlines()

        else:
            self.white_list= open('/memory/wangqimei/train_data_path/Piccolo_pair_5fold/WL/%d/Testset/images.txt' % split_id).readlines()
            self.nbi_list = open('/memory/wangqimei/train_data_path/Piccolo_pair_5fold/NBI/%d/Testset/images.txt' % split_id).readlines()

        self.white_list = list(map(lambda x: x.strip(), self.white_list))
        self.nbi_list = list(map(lambda x: x.strip(), self.nbi_list))
        char_to_num = {
            'Hyperplasia': 0,
            'Adenoma': 1,
            'Adenocarcinoma': 2
        }
        self.white_label = [self.label_dict[item.split('/')[-1].split('_')[0].lstrip('0')] for item in self.white_list]
        self.white_label =list(map(lambda x: char_to_num[x], self.white_label))
        self.white_label = np.array(self.white_label, dtype=np.int8)
        self.nbi_label = [self.label_dict[item.split('/')[-1].split('_')[0].lstrip('0')] for item in self.nbi_list]
        self.nbi_label =list(map(lambda x: char_to_num[x], self.nbi_label))
        self.nbi_label = np.array(self.nbi_label, dtype=np.int8)
        unique, counts = np.unique(self.white_label, return_counts=True)
        print('wli:', dict(zip(unique, counts)))

    def __getitem__(self, white_index):
        wht_path = self.white_list[white_index]
        nbi_path = self.nbi_list[white_index]

        white_img = imageio.imread(wht_path)
        nbi_img = imageio.imread(nbi_path)

        # augmentation
        if self.is_train and self.enable_aug:
            angle = random.randint(-180, 180)
            white_img = rotate(white_img, angle)
            nbi_img = rotate(nbi_img, angle)

            if random.random() > 0.5:
                nbi_img = np.flip(nbi_img, 0)
                white_img = np.flip(white_img, 0)

            if random.random() > 0.5:
                nbi_img = np.flip(nbi_img, 1)
                white_img = np.flip(white_img, 1)

        # resize
        white_img = resize(white_img, (448, 448))
        nbi_img = resize(nbi_img, (448, 448))

        # swap axis
        white_img = np.swapaxes(white_img, 0,-1)
        nbi_img = np.swapaxes(nbi_img, 0,-1)

        label = self.white_label[white_index]

        return white_img, nbi_img, label, wht_path,nbi_path

    def __len__(self):
        return len(self.white_list)

