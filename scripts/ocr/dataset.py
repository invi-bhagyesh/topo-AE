import os
import torch
import cv2
import torch.utils.data
from torch.utils.data import Dataset    

class test_adv_dataset(Dataset):
    def __init__(self, height, width, img_path):      
        self.height = height
        self.width = width
        self.img_path = img_path
        self.dataset = []
        
        img = [] 
        for i,j,k in os.walk(self.img_path):            
            for file in k:
                file_name = os.path.join(i ,file)
                img.append(file_name)
        self.total_img_name = img
        
        for img_name in self.total_img_name:
            base_name = os.path.basename(img_name)
            parts = base_name.rsplit('_', 2)
            if len(parts) == 3:
                img_index, label, img_adv = parts
                img_adv = img_adv.split('.')  # split extension
                index_or_advlogo = img_adv[0]
                self.dataset.append([img_name, label, img_index, index_or_advlogo])
            else:
                print(f"Skipping unexpected filename format: {base_name}")
        self.dataset = sorted(self.dataset)

    def __getitem__(self, index):
        img_name, label, img_index, index_or_advlogo = self.dataset[index]        
        IMG = cv2.imread(img_name) 
        IMG = cv2.resize(IMG, (self.width, self.height))
        
        # binarization processing
        gray = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_b = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        img_b = torch.FloatTensor(img_b)
        img_b = img_b / 255 # normalization to [0,1]
        img_b = img_b.permute(2,0,1) # [C, H, W]

        img = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
        img = torch.FloatTensor(img)
        img = img /255 # normalization to [0,1]
        img = img.permute(2,0,1) # [C, H, W]

        return img_b, img, label, img_index, index_or_advlogo, img_name

    def __len__(self):
        return len(self.dataset)
