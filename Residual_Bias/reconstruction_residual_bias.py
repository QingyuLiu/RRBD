import argparse
import copy
from datetime import datetime
import glob
import json
import os
import random

import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from albumentations import Compose, HorizontalFlip, ImageCompression, GaussNoise, GaussianBlur,Resize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
import torch
from pathlib import Path
from reconstruction import ddim_loop_target_step


class ImageDataset(Dataset):
    def __init__(self, root_dir,vae,name =  "image",train = True):
        self.root_dir = root_dir
        self.vae = vae
        self.train = train
        if os.path.isfile(root_dir):
            source_files_list = []
            labels = []
            with open(root_dir, 'r', encoding='utf-8') as f:
                content = f.readlines()

                for item in content:
                    item = json.loads(item)
                    source_files_list.append(item[name])
                    labels.append(item["label"])
            self.labels = labels


        self.dataset = source_files_list
        self.target_size = 512

        self.aug = self.create_train_aug()
        self.transform = self.transform_all()

    def __len__(self):
        return len(self.dataset)

    def create_train_aug(self):
        return Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.2),
            GaussianBlur(blur_limit=3, p=0.2),
            HorizontalFlip(),
        ]
        )
    def transform_all(self):
        return Compose([Resize(p=1, height=200, width=200),
                        Resize(p=1, height=self.target_size, width=self.target_size)]
        )

    def __getitem__(self, idx):
        img = cv2.imread(self.dataset[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] < 200:
            data = Resize(p=1, height=self.target_size, width=self.target_size)(image=img)
        else:
            data = self.transform(image=img)
        # only training need
        if self.train:
            data = self.aug(image=data["image"])

        img = data["image"]
        x = transforms.ToTensor()(img)[None, ...].to(device)
        x = 2. * x - 1.
        posterior = self.vae.encode(x).latent_dist
        latents = posterior.mean * 0.18215
        return latents[0], x[0] ,self.labels[idx],self.dataset[idx]


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents



def latent_to_img(pipe, latent):
    # pipe.vae.requires_grad_(False)
    latents = 1 / pipe.vae.config.scaling_factor * latent
    image = pipe.vae.decode(latents, return_dict=False)[0]
    # image = (image / 2 + 0.5).clamp(0, 1)
    # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    # image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return image

def save_difference(input_imgs,theory,diff_latent,output_dir,train=False):
    differece_imgs_name = []
    aug_img_name = []



    for i in range(len(input_imgs)):
        eta = abs(diff_latent[i] - theory[i]).cpu().detach()
        timestamp = datetime.now()


        if train:
            name = f"{output_dir}/aug_img/{timestamp}.png"
            aug_img_name.append(name)
            file_path = Path(f"{name}")
            aug_input_img = ((input_imgs[i] / 2 + 0.5).clamp(0, 1) * 255.0).to(torch.uint8)
            aug_input_img = aug_input_img.permute(1, 2, 0)
            aug_input_img = aug_input_img.contiguous()
            aug_input_img = aug_input_img.detach().cpu().numpy()
            aug_input_img = Image.fromarray(aug_input_img)
            # aug_input_img = transforms.Resize(224)(aug_input_img)
            aug_input_img.save(file_path)


        name = f"{output_dir}/residual_bias/{timestamp}.npz"
        differece_imgs_name.append(name)
        np.savez(name,eta)

    return differece_imgs_name,aug_img_name

def save_file_names_labels(file_names,aug_img_names,labels,origin_images,output_dir):
    with open(f"{output_dir}/train_difference_image_labels.jsonl", 'w', encoding='utf-8') as test_f:
        if len(aug_img_names) == 0:
            aug_img_names = origin_images
        json_lines=[]
        for i,name in enumerate(file_names):
            data_entry = {"residual_bias": name, "aug_img":aug_img_names[i],"origin_image":origin_images[i],"label": labels[i].item()}
            json_line = json.dumps(data_entry, ensure_ascii=False)
            json_lines.append(json_line+"\n")
        test_f.writelines(json_lines)
def reconstruction_residual_bias(pipe, data_loader,output_dir, device, train = False):
    file_names = []
    file_labels = []
    origin_files = []
    aug_img_names = []
    for (latents,input_img,labels,image_names) in tqdm(data_loader):
        file_labels+=list(labels)
        origin_files+=list(image_names)
        prompt = [""] * latents.shape[0]
        latents = latents.to(device=device)

        reconstruct_latent,theory = ddim_loop_target_step(pipe, prompt, latents, 0, device=device, num_inference_steps=50,
                                                   guidance_scale=1)

        inputs_name,aug_img_name = save_difference(input_img,theory,abs(latents - reconstruct_latent),output_dir, train=train)

        file_names+=inputs_name
        aug_img_names+=aug_img_name

    save_file_names_labels(file_names,aug_img_names,file_labels,origin_files,output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_train_sd", default=None, required=True, type=str)
    parser.add_argument("--pre_train_G_LDM", default=None, required=True,type=str)
    parser.add_argument("--batch_size", default=1,type=int)
    parser.add_argument("--device",default="1",type=str)
    parser.add_argument("--data_json", default=None, required=True,type=str)
    parser.add_argument("--train", default=False, type=bool)
    parser.add_argument("--diff_save_path", default=None, required=True, type=str)
    args = parser.parse_args()

    data_json = args.data_json
    train = args.train
    diff_save_path = args.diff_save_path
    device = f"cuda:{args.device}"
    batch_size = args.batch_size

    unet = UNet2DConditionModel.from_pretrained(args.pre_train_G_LDM, subfolder="unet")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pre_train_sd, unet=unet  , safety_checker=None
    )

    device = torch.device(device)
    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)


    dataset = ImageDataset(data_json, pipe.vae,train=train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    os.makedirs(f"{diff_save_path}/residual_bias", exist_ok=True)
    os.makedirs(f"{diff_save_path}/aug_img", exist_ok=True)
    reconstruction_residual_bias(pipe, dataloader, diff_save_path, device, train = train)

