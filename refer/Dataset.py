from torch.utils.data import Dataset
from torchvision import transforms
from refer.img_processor import getfilelist
import os
from PIL import Image

class Dataset(Dataset):
    def __init__(self, folderpath, transform=None):
        self.folderpath = folderpath
        self.transform = transform
        self.filename_list = getfilelist(folderpath)
        self.imglabel={"fd":1,"fs":2,"md":3,"ms":4,"none":0}
    def __len__(self):
        return len(self.filename_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folderpath, self.filename_list[idx])
        img = Image.open(img_path)
        
        label = self.imglabel[self.filename_list[idx].split('_')[0]]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


import os
from PIL import Image

class KinshipDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        Args:
            root_dirs (list): 包含多个数据集文件夹路径的列表。
            transform (callable, optional): 对图像应用的变换。
        """
        self.root_dirs = root_dirs
        self.transform = transform

        # 定义正样本（亲缘关系）的文件夹路径
        self.kin_folders = ['father-dau', 'father-son', 'mother-dau', 'mother-son', 'nonkin']
        self.imglabel = {"nonkin": 0,"fd": 1, "fs": 2, "md": 3, "ms": 4, "bb": 5, "ss": 6, "sibs": 7}
        self.short_relation = {
            'father-dau': 'fd',
            'father-son': 'fs',
            'mother-dau': 'md',
            'mother-son': 'ms',
            'nonkin': 'nonkin'
        }

        # 获取所有文件夹中的图像对
        self.kin_pairs = self.get_all_kpairs()

    def get_kpairs(self, root_dir):
        """获取单个数据集中的亲缘关系图像对 (img1, img2, label)"""
        pairs = []
        for folder in self.kin_folders:
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                continue  # 跳过不存在的文件夹
            images = sorted(os.listdir(folder_path))
            for i in range(0, len(images) - 1, 2):  # 假设图片成对出现
                if images[i].lower() == 'thumbs.db' or images[i + 1].lower() == 'thumbs.db':
                    continue  # 忽略 Thumbs.db 文件
                img1_path = os.path.join(folder_path, images[i])
                img2_path = os.path.join(folder_path, images[i + 1])
                if folder in self.short_relation:
                    pairs.append((img1_path, img2_path, self.imglabel[self.short_relation[folder]]))
        return pairs

    def get_all_kpairs(self):
        """从多个数据集文件夹中获取所有亲缘关系图像对"""
        all_pairs = []
        for root_dir in self.root_dirs:
            all_pairs.extend(self.get_kpairs(root_dir))
        return all_pairs

    def __len__(self):
        return len(self.kin_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.kin_pairs[idx]

        # 加载图片
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # 应用变换
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), label

# 使用示例
# img_folder1 = 'data\\KinFaceW-I\\images'
# img_folder2 = 'data\\KinFaceW-II\\images'
# dataset = KinshipDataset(root_dirs=[img_folder1, img_folder2])
