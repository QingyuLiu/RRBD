# -*- coding: utf-8 -*-
import base64
import json
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None,base64=False):
        self.root_dir = root_dir
        # self.transform = transform
        # List all image files in the folder
        if os.path.isdir(root_dir):
            image_files = [os.path.join(root_dir,f) for f in os.listdir(root_dir) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        else:
            dataset = load_dataset('json', data_files=root_dir)
            image_files = dataset["train"]["origin_image"]
        self.dataset = image_files
        self.base64 = base64
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # image = np.array(Image.open(self.dataset[idx]))
        if self.base64:
            with open(self.dataset[idx],'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            return image_data,self.dataset[idx]
        return self.dataset[idx]



def image_caption(image_folder,output_file,model_path,batch_size=1,device = "cuda:0"):
    processor = LlavaNextProcessor.from_pretrained(model_path)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(device)

        # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image")
    conversation = [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": "image caption or prompt"},
              {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # prompt = f'<|user|>\n<image>\n image caption \n<|assistant|>\n'
    # prompts = [prompt]*batch_size
    # prompts = processor(conversation, add_generation_prompt=True)

    dataset = ImageDataset(image_folder)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # Save data to JSON Lines file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Generate captions for each image
        for image_paths in tqdm(dataloader):
            # images, image_paths = data["img"], data["path"]
            images = [Image.open(path) for path in image_paths]
            prompts = [prompt]*len(image_paths)
            try:
                inputs = processor(prompts, images, return_tensors="pt").to(device,torch.float16)
                # autoregressively complete prompt
                outputS = model.generate(**inputs, max_new_tokens=100)

                for i in range(len(image_paths)):
                    output = processor.decode(outputS[i], skip_special_tokens=True)
                    caption = output.split('[/INST]')[-1]
                    caption = caption.replace("\"","")
                    data_entry = {"file_path": image_paths[i], "caption": caption}
                    json_line = json.dumps(data_entry, ensure_ascii=False)
                    f.write(json_line + '\n')
                    # data.append(data_entry)
            except:
                print(f"Error in image: {image_paths}")



    print(f"Captions generated and saved to {output_file}")

image_folder = "dataset_json/SD_1.5.jsonl"
output_dir = f"dataset_json/SD_1.5_caption.jsonl"
model_path = "model/llava-v1.6-mistral-7b-hf"
print(image_folder)
image_caption(image_folder,output_dir,model_path,batch_size=15,device = "cuda:0")