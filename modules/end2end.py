import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import open_clip
import pandas as pd
import wandb
import copy
import torch
import torchvision
import torch.nn as nn
from torch import optim
import torchxrayvision as xrv
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
from models import DenseNetE2E, ViTE2E

random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class ImageDataset():
    def __init__(self, class2images, preprocess, image_dir, class2label):
        # Initialize image paths and corresponding texts
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
        self.images = []
        
        for class_name, images in class2images.items():
            for image in images:
                self.image_paths.append(f"{image_dir}{image}")
                self.labels.append(class2label[class_name])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_paths[idx]))
        label = self.labels[idx]
        return image, label


transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
def densenet_preprocess(image):
    image = image.convert("RGB")
    img = np.array(image)
    img = xrv.datasets.normalize(img, 255)
    img = img.mean(2)[None, ...]
    img = transform(img)
    img = torch.from_numpy(img)
    return img


def train_model(model, train_loader, val_loader, num_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -float("inf")
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.type(torch.float32).to(device)
            labels = labels.type(torch.LongTensor).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": epoch * len(train_loader) + i})

        val_acc = eval_model(model, val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader)}, Val Acc: {val_acc}")
        wandb.log({"val_acc": val_acc, "epoch": epoch})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

    return best_model


def eval_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    
    return accuracy


def run_dataset(args, dataset_name, base_model, preprocess):
    wandb.init(project="e2e_classification", 
               name=f"{args.modality}_{args.model_name}_{dataset_name}_{args.batch_size}_{args.num_epochs}_{args.lr}",
               config={
                "modality": args.modality,
                "batch_size": args.batch_size,
                "epochs": args.num_epochs,
                "model_name": args.model_name,
                "dataset_name": dataset_name,
                "lr": args.lr}
                )
                
    print(f"Running {args.modality} {args.model_name} on {dataset_name}")
    dataset_dir = f"./data/datasets/{dataset_name}"
    class2images_train = pickle.load(open(f"{dataset_dir}/splits/class2images_train.p", "rb"))
    class2images_val = pickle.load(open(f"{dataset_dir}/splits/class2images_val.p", "rb"))
    class2images_test = pickle.load(open(f"{dataset_dir}/splits/class2images_test.p", "rb"))

    class2label = {class_name: i for i, class_name in enumerate(class2images_train.keys())}

    dataset_train = ImageDataset(class2images_train, preprocess, args.image_dir, class2label)
    dataset_test = ImageDataset(class2images_test, preprocess, args.image_dir, class2label)
    dataset_val = ImageDataset(class2images_val, preprocess, args.image_dir, class2label)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.model_name == "vit": model = ViTE2E(base_model, len(class2label))
    elif args.model_name == "densenet": model = DenseNetE2E(base_model, len(class2label))
    model.to(device)

    best_model = train_model(model, train_loader, val_loader, args.num_epochs, args.lr)

    # Evaluate the model
    val_acc = eval_model(best_model, val_loader)
    ood_acc = eval_model(best_model, test_loader)

    average_acc = round((val_acc + ood_acc) / 2, 2)
    gap = round(abs(val_acc - ood_acc), 2)

    print(f"Ind Acc: {val_acc}, OOD Acc: {ood_acc}, Gap: {gap}, Average: {average_acc}")

    # close wandb
    wandb.finish()
    return val_acc, ood_acc, gap, average_acc, best_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--modality", type=str, default="xray, skin")
    parser.add_argument("--model_name", type=str, default="vit, densenet")
    parser.add_argument("--image_dir", type=str, default="./data/datasets/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()

    if args.model_name == "vit":
        if args.modality == "xray":
            base_model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whyxrayclip")
        elif args.modality == "skin":
            base_model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whylesionclip")

    elif args.model_name == "densenet":
        preprocess = densenet_preprocess
        if args.modality == "xray":
            base_model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb")
        elif args.modality == "skin":
            model = DenseNetE2E(xrv.models.DenseNet(), 9)
            weight = torch.load(f"./data/model_weights/densenet_skin.pt")
            model.load_state_dict(weight)
            base_model = model.denset_model
    
    if args.modality == "xray":
        dataset_lists = ["NIH-sex", "NIH-age", "NIH-pos", "CheXpert-race", "NIH-CheXpert", "pneumonia", "COVID-QU", "NIH-CXR", "open-i", "vindr-cxr"]
    elif args.modality == "skin":
        dataset_lists = ["ISIC-sex", "ISIC-age", "ISIC-site", "ISIC-color", "ISIC-hospital", "HAM10000", "BCN20000", "PAD-UFES-20", "Melanoma", "UWaterloo"]
    
    results_dict = {}
    for dataset_name in dataset_lists:
        ind_acc, out_acc, gap, avg, best_model = run_dataset(args, dataset_name, base_model, preprocess)
        results_dict[dataset_name] = {"ind_acc": ind_acc, "out_acc": out_acc, "gap": gap, "avg": avg}
    
    # reshape to one row
    csv_df = pd.DataFrame.from_dict(results_dict).T

    # Reshape the data
    reshaped_df = pd.DataFrame(csv_df.values.flatten()).T

    # Create new column names
    new_columns = [f"{row_label}_{col_label}" for row_label in csv_df.index for col_label in csv_df.columns]

    # Assign new column names to reshaped dataframe
    reshaped_df.columns = new_columns

    reshaped_df.to_csv(f"./data/results/end2end_{args.model_name}_{args.modality}.csv")