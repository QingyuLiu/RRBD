import argparse
from datetime import datetime
import json
import os

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
from reconstruction import ddim_loop_target_step


class ImageDataset(Dataset):
    def __init__(self, root_dir,vae,name =  "image",train = True):
        self.root_dir = root_dir
        self.vae = vae
        self.train = train
        source_files_list = []
        if os.path.isfile(root_dir):

            labels = []
            with open(root_dir, 'r', encoding='utf-8') as f:
                data = json.load(f)

                for item in data:
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
            # GaussNoise(p=0.2),
            GaussianBlur(blur_limit=3, p=0.2),
            HorizontalFlip(),
        ]
        )
    def transform_all(self):
        return Compose([
            Resize(p=1, height=self.target_size, width=self.target_size)]
        )

    def __getitem__(self, idx):
        img = cv2.imread(self.dataset[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    latents = 1 / pipe.vae.config.scaling_factor * latent
    image = pipe.vae.decode(latents, return_dict=False)[0]
    # image = (image / 2 + 0.5).clamp(0, 1)
    # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    # image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return image

def save_difference(input_imgs,theory,latents,reconstruct_latent,output_dir,train=False):
    residual_bias_name = []
    differece_imgs_name = []
    transform_resize = transforms.Resize((224, 224))
    for i in range(len(input_imgs)):

        timestamp = datetime.now()
        eta = abs((latents[i]-reconstruct_latent[i]) - theory[i])
        name = f"{output_dir}/residual_bias_latent/{timestamp}.npz"
        np.savez(name, eta.cpu().detach())
        residual_bias_name.append(name)

        theory_img = latent_to_img(pipe, (theory[i]*(10 ** 3)).unsqueeze(0))[0]
        recons_img = latent_to_img(pipe, reconstruct_latent[i].unsqueeze(0))[0]
        measured_img = input_imgs[i] - recons_img
        eta_img = abs(measured_img-theory_img)
        eta_img = preprocess_img(eta_img, transform_resize)
        name = f"{output_dir}/residual_bias_rgb/{timestamp}.png"
        differece_imgs_name.append(name)
        eta_img.save(name)


    return residual_bias_name,differece_imgs_name
import torch.fft

def to_frequency_domain(residual):
    fft = torch.fft.fft2(residual)
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shifted)
    log_magnitude = torch.log1p(magnitude)
    center_crop = log_magnitude[ :, 144:368, 144:368]
    return center_crop

def preprocess_img(difference,transform_resize):
    difference = ((difference/2).clamp(0, 1) * 255.0).to(torch.uint8)
    difference = difference.permute(1, 2, 0).contiguous().detach().cpu().numpy()
    difference = Image.fromarray(difference)
    difference = transform_resize(difference)
    return difference

def save_file_names_labels(residual_bias_name,differece_imgs_name,labels,origin_images,output_dir):
    with open(f"{output_dir}/train_difference_image_labels.jsonl", 'a', encoding='utf-8') as test_f:
        json_lines=[]
        for i in range(len(labels)):
            data_entry = {"residual_bias_latent": residual_bias_name[i], "residual_bias_rgb": differece_imgs_name[i],
                         "origin_image":origin_images[i],"label": labels[i].item()}
            json_line = json.dumps(data_entry, ensure_ascii=False)
            json_lines.append(json_line+"\n")
        test_f.writelines(json_lines)
def reconstruction_residual_bias(pipe, data_loader,output_dir, device, train = False):

    with torch.no_grad():
        for (latents,input_img,labels,image_names) in tqdm(data_loader):
            prompt = [""] * latents.shape[0]
            latents = latents.to(device=device)

            reconstruct_latent,theory = ddim_loop_target_step(pipe, prompt, latents, 0, device=device, num_inference_steps=50,
                                                       guidance_scale=1)

            residual_bias_name, differece_imgs_name = save_difference(
                input_img, theory, latents, reconstruct_latent, output_dir, train=train)

            save_file_names_labels(residual_bias_name, differece_imgs_name, list(labels), list(image_names), output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_train_sd", default="stabilityai/stable-diffusion-2-1-base", required=True, type=str)
    parser.add_argument("--pre_train_G_LDM", default="checkpoint", required=True, type=str)
    parser.add_argument("--batch_size", default=120,type=int)
    parser.add_argument("--device",default="0",type=str)
    parser.add_argument("--data_json", default=None, required=True, type=str)
    parser.add_argument("--train", default=False, type=bool)
    parser.add_argument("--diff_save_path", default="output",required=True, type=str)
    args = parser.parse_args()

    data_json = args.data_json
    train = args.train
    diff_save_path = args.diff_save_path
    device = f"cuda:{args.device}"
    batch_size = args.batch_size

    unet = UNet2DConditionModel.from_pretrained(args.pre_train_G_LDM, subfolder="unet_ema")
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
    os.makedirs(f"{diff_save_path}", exist_ok=True)
    os.makedirs(f"{diff_save_path}/residual_bias_rgb", exist_ok=True)
    os.makedirs(f"{diff_save_path}/residual_bias_latent", exist_ok=True)

    reconstruction_residual_bias(pipe, dataloader, diff_save_path, device, train = train)

