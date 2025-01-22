import argparse
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import calculate_metrics
import torch.nn.functional as F
from two_stream_model.two_resnet_attention import Two_Stream_Net

def prepare_data(dataset_path,batch_size=32,scaler_filename=None):
    dataset = load_dataset('json', data_files=dataset_path)
    print(dataset)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    column_names = dataset["train"].column_names
    print(column_names)
    classes = list(set(dataset["train"]["label"]))


    def standard_dataset(filename):
        feature_list = []
        for feature_file_path in dataset['train']["residual_bias"]:
            feature_list.append(np.load(feature_file_path)["arr_0"])
        X = np.array(feature_list)

        # StandardScaler
        size = X.shape
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X.reshape(size[0], -1))

        joblib.dump(scaler, filename)
    if scaler_filename is None:
        scaler_filename = f'{dataset_path[:-6]}_scaler.pkl'
    else:
        scaler_filename = f'{scaler_filename[:-6]}_scaler.pkl'

    if not os.path.exists(scaler_filename):
        scaler = standard_dataset(scaler_filename)
    else:
        scaler = joblib.load(scaler_filename)

    def preprocess(examples):
        examples["residual_bias"] = [np.load(image_path)["arr_0"] for image_path in examples['residual_bias']]
        examples["residual_bias"] = np.array(examples["residual_bias"])
        size = examples["residual_bias"].shape
        examples["residual_bias"] = scaler.transform(examples["residual_bias"].reshape(size[0], -1))
        examples["residual_bias"]= torch.Tensor(examples["residual_bias"]).reshape(size)

        # images = [image.convert("RGB") for image in examples[image_column]]
        rgb_images = [Image.open(image_path).convert("RGB") for image_path in examples['aug_img']]
        examples["aug_img"] = [transform(image) for image in rgb_images]
        # examples["label"] = examples['label']
        return examples




    # dataset["train"] = dataset["train"].shuffle()
    dataset = dataset["train"].with_transform(preprocess)



    def collate_fn(examples):
        pixel_values = torch.stack([example["residual_bias"] for example in examples])
        frequency_image = torch.stack([example["aug_img"] for example in examples])
        # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        labels = torch.as_tensor([example["label"] for example in examples], dtype=torch.long)
        return {"residual_bias": pixel_values, "label": labels,"aug_img":frequency_image}


    # DataLoaders creation:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=0,
    )
    return dataloader,classes
def create_model(num_classes):
    model = Two_Stream_Net(num_classes = num_classes)
    return model

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10,save_path = "train"):

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x in tqdm(train_loader):

            inputs, labels = x["residual_bias"].to(device), x["label"].to(device)
            rgb_image = x["aug_img"].to(device)

            optimizer.zero_grad()
            _,outputs = model(inputs,rgb_image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)



        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_path}/two-stream_{epoch}.pt")
        epoch_loss = running_loss / len(train_loader.dataset)
        print('-' * 30)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        evaluate_model(model, test_loader, device)
    torch.save(model.state_dict(), f"{save_path}/two-stream_last.pt")

def evaluate_model(model, val_loader, device):
    model.eval()
    true_ = []
    pred_ = []

    with torch.no_grad():
        for x in val_loader:
            inputs, labels = x["residual_bias"].to(device), x["label"].to(device)
            rgb_image = x["aug_img"].to(device)

            _,outputs = model(inputs,rgb_image)
            softmax_outputs = F.softmax(outputs, dim=1)


            one_hot_labels = F.one_hot(labels, num_classes=softmax_outputs.shape[1])
            true_.extend(one_hot_labels.cpu().numpy())
            pred_.extend(softmax_outputs.cpu().numpy())

    ground_truth, prediction = np.array(true_), np.array(pred_)
    prediction_binary = np.array([prediction[:, 0], np.sum(prediction[:, 1:], axis=1)]).T
    ground_truth_binary = np.array([ground_truth[:, 0], np.sum(ground_truth[:, 1:], axis=1)]).T

    _, pred_labels = torch.max(torch.Tensor(prediction_binary), 1)
    _, true_labels = torch.max(torch.Tensor(ground_truth_binary), 1)

    TPR,FPR, FDR,  ACC,AUPRC,AUROC,BDR,eer = calculate_metrics(true_labels, prediction_binary[:, 1])
    print("Binary-class:")
    print((f"AUPRC: {AUPRC}"))
    print((f"AUROC: {AUROC}"))
    print((f"BDR: {BDR}"))
    print((f"EER: {eer}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_json",
                        default="data_json/train_difference_image_labels.jsonl",
                        type=str)
    parser.add_argument("--test_data_json",
                        default="data_json/test_difference_image_labels.jsonl",
                        type=str)
    parser.add_argument("--save_path",
                        default="detector",
                        type=str)
    parser.add_argument("--device",
                        default="1",
                        type=str)
    parser.add_argument("--train_batch_size", default=256, type=int)
    parser.add_argument("--test_batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    args = parser.parse_args()

    train_dir = args.train_data_json
    test_dir = args.test_data_json
    save_path = args.save_path
    device = f"cuda:{args.device}"
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    train_loader,classes= prepare_data(train_dir,batch_size=train_batch_size)
    model = create_model(len(classes))

    device = torch.device(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    test_loader, _ = prepare_data(test_dir, batch_size=test_batch_size, scaler_filename=train_dir)
    os.makedirs(save_path, exist_ok=True)
    model.to(device)

    train_model(model, train_loader, test_loader,criterion, optimizer, device, num_epochs,save_path=save_path)
    evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    main()
