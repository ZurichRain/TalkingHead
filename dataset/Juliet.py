import torch.nn as nn
import torch
from torch.utils.data import Dataset
import os
from random import random
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms

def collate_fn(examples):
    last_frame_image = torch.stack([example["last_frame_image"] for example in examples])
    last_frame_image = last_frame_image.to(memory_format=torch.contiguous_format).float()
    # input_ids = torch.stack([example["input_ids"] for example in examples])
    cur_frame_image = torch.stack([example["cur_frame_image"] for example in examples])
    cur_frame_image = cur_frame_image.to(memory_format=torch.contiguous_format).float()
    return {"last_frame_image": last_frame_image, "cur_frame_image": cur_frame_image}
    # return {"last_frame_image": last_frame_image}

class JulietData(Dataset):
    '''
        dataformat:
            lastframeid curframeid
    '''

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        
        self.data_name = args.data.data_name
        print(self.data_name)
        self.data_paths = args.data.txt_file # txt_file 是存储
        self.data_root = args.data.data_root
        self.image_paths = []
        self.lastImgPath = []
        self.curImgPath = []

        for data_name in self.data_name.split("+"):
            Curdatapath = self.data_root + data_name + '/data.txt'
            with open(Curdatapath, "r") as f:
                img_p = f.read().splitlines()
                self.image_paths+=img_p
                self.lastImgPath += [self.data_root + data_name + '/Image/' + line.split(' ')[0] for line in img_p]
                self.curImgPath += [self.data_root + data_name + '/Image/' + line.split(' ')[1] for line in img_p]
        self._length = len(self.image_paths)
        # image_list_path = os.path.join(self.data_root, 'data.txt')
        # with open(image_list_path, "r") as f:
        #     self.image_num = f.read().splitlines()

        # self.labels = {
        #     "last_frame_id": [int(l.split('_')[0]) for l in self.image_paths],
        #     "image_path_": [os.path.join(self.data_root, 'images', l+'.jpg') for l in self.image_paths],
        #     "audio_smooth_path_": [os.path.join(self.data_root, 'audio_smooth', l + '.npy') for l in self.image_paths],
        #     "landmark_path_": [os.path.join(self.data_root, 'landmarks', l+'.lms') for l in self.image_paths],
        #     "reference_path": [l.split('_')[0] + '_' + str(random.choice(list(set(range(1, int(self.image_num[int(l.split('_')[0])-1].split()[1])))-set(range(int(l.split('_')[1])-60, int(l.split('_')[1])+60)))))
        #                        for l in self.image_paths],
        # }

        # self.labels = {
        #     "last_frame_image_path": [os.path.join(self.data_root, 'Image', l.split(' ')[0]) for l in self.image_paths],
        #     "cur_frame_image_path": [os.path.join(self.data_root, 'Image', l.split(' ')[1]) for l in self.image_paths],
        # }

        self.labels = {
            "last_frame_image_path": self.lastImgPath,
            "cur_frame_image_path": self.curImgPath,
        }

        self.size = args.data.size
        interpolation = "bicubic"
        self.interpolation = {
                                "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        # self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(args.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.data.resolution) if args.data.center_crop else transforms.RandomCrop(args.data.resolution),
                transforms.RandomHorizontalFlip() if args.data.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # 224

        self.train_transforms_last = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224) if args.data.center_crop else transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip() if args.data.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __len__(self):
        return self._length
    
    
    

    def get_img(self, image_path):
        # default to score-sde preprocessing
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        # examples["pixel_values"] = [train_transforms(image) for image in images]
        image = self.train_transforms(image)
        # image = np.array(image).astype(np.uint8)
        # image = Image.fromarray(image)
        # h, w = image.size
        # if self.size is not None:
        #     image = image.resize((self.size, self.size), resample=self.interpolation)

        # image = np.array(image).astype(np.uint8)
        # image = (image / 127.5 - 1.0).astype(np.float32)
        return image
    
    def get_img_last(self, image_path):
        # default to score-sde preprocessing
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        # examples["pixel_values"] = [train_transforms(image) for image in images]
        image = self.train_transforms_last(image)
        # image = np.array(image).astype(np.uint8)
        # image = Image.fromarray(image)
        # h, w = image.size
        # if self.size is not None:
        #     image = image.resize((self.size, self.size), resample=self.interpolation)

        # image = np.array(image).astype(np.uint8)
        # image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, idx):
        example = dict((k, self.labels[k][idx]) for k in self.labels)

        example["last_frame_image"] = self.get_img_last(example["last_frame_image_path"])
        example["cur_frame_image"] = self.get_img(example["cur_frame_image_path"])


        # last_frame_image = Image.open(example["last_frame_image_path"])
        # if not last_frame_image.mode == "RGB":
        #     last_frame_image = last_frame_image.convert("RGB")

        # # default to score-sde preprocessing
        # last_frame_image = np.array(last_frame_image).astype(np.uint8)
        # last_frame_image = Image.fromarray(last_frame_image)
        # h, w = last_frame_image.size
        # if self.size is not None:
        #     last_frame_image = last_frame_image.resize((self.size, self.size), resample=self.interpolation)

        # last_frame_image = np.array(last_frame_image).astype(np.uint8)
        # example["last_frame_image"] = (last_frame_image / 127.5 - 1.0).astype(np.float32)

        # landmarks = np.loadtxt(example["landmark_path_"], dtype=np.float32)
        # landmarks_img = landmarks[13:48]
        # landmarks_img2 = landmarks[0:4]
        # landmarks_img = np.concatenate((landmarks_img2, landmarks_img))
        # scaler = h / self.size
        # example["landmarks"] = (landmarks_img / scaler)

        # mask
        # mask = np.ones((self.size, self.size))
        # mask[(landmarks[30][1] / scaler).astype(int):, :] = 0.
        # mask = mask[..., None]
        # image_mask = (image * mask).astype(np.uint8)
        # example["image_mask"] = (image_mask / 127.5 - 1.0).astype(np.float32)

        # example["audio_smooth"] = np.load(example["audio_smooth_path_"]).astype(np.float32)

        #add for reference
        # image_r = Image.open(os.path.join(self.data_root, 'images', example["reference_path"] +'.jpg'))
        # if not image_r.mode == "RGB":
        #     image_r = image_r.convert("RGB")

        # img_r = np.array(image_r).astype(np.uint8)
        # image_r = Image.fromarray(img_r)
        # image_r = image_r.resize((self.size, self.size), resample=self.interpolation)
        # image_r = np.array(image_r).astype(np.uint8)
        # example["reference_img"] = (image_r / 127.5 - 1.0).astype(np.float32)

        return example


    