import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
import torchxrayvision as xrv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import clip
import open_clip
from utils import *
from models import DenseNetE2E
# from medclip import MedCLIPProcessor, MedCLIPModel, MedCLIPVisionModelViT # Please refer to the original MedCLIP repository to set up the environment: https://github.com/RyanWangZf/MedCLIP
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel

from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device being used:", device)


class WhyMedClip(Dataset):

    def __init__(self, data, preprocess):
        self.data = data
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image_id = image_path.split("/")[-1].split(".")[0]
        image = self.preprocess(open_image(image_path))
        return (image_path, image)
    

class PixelFeatureExtractor:
    def __init__(self):
        pass

    def encode_image(self, images):
        features = [self.extract_features(image) for image in images]
        return np.array(features)

    def extract_features(self, image):
        image_resized = image.resize((28, 28), Image.ANTIALIAS)
        image_gray = image_resized.convert('L')
        image_array = np.array(image_gray)
        image_flattened = image_array.flatten()
        return image_flattened
    
    def preprocess(self, image):
        return image


class MedCLIPFeatureExtractor:
    def __init__(self):
        self.processor = MedCLIPProcessor()
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
        model.cuda()
        self.model = model
    
    def encode_image(self, images):
        inputs = self.processor(images=images, text=["dummy text"], return_tensors="pt", padding=True)
        image_features = self.model(**inputs)["img_embeds"]
        return image_features
    
    def preprocess(self, image):
        return image
    

class PubMedFeatureExtractor:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        model.cuda()
        self.model = model
    
    def encode_image(self, images):
        inputs = self.processor(images=images, text=["dummy text"], return_tensors="pt", padding=True).to(device)
        image_features = self.model(**inputs)["image_embeds"]
        
        return image_features
    
    def preprocess(self, image):
        return image


class DenseNetXrayFeatureExtractor:
    def __init__(self):
        self.model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb") 
        self.transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
        self.model.to(device)
    
    def encode_image(self, images):
        image_features = self.model.features2(images)
        return image_features
    
    def preprocess(self, image):
        # convert to RGB
        image = image.convert("RGB")
        img = np.array(image)
        img = xrv.datasets.normalize(img, 255)
        img = img.mean(2)[None, ...]
        img = self.transform(img)
        img = torch.from_numpy(img)
        return img


class DenseNetSkinFeatureExtractor:
    def __init__(self):
        base_model = xrv.models.DenseNet()
        self.model = DenseNetE2E(base_model, 9)
        weight = torch.load("./data/model_weights/densenet_skin.pt")
        self.model.load_state_dict(weight)
        self.transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
        self.model.to(device)
    
    def encode_image(self, images):
        image_features = self.model.denset_model.features2(images)
        return image_features
    
    def preprocess(self, image):
        # convert to RGB
        image = image.convert("RGB")
        img = np.array(image)
        img = xrv.datasets.normalize(img, 255)
        img = img.mean(2)[None, ...]
        img = self.transform(img)
        img = torch.from_numpy(img)
        return img


def load_model(model_name):
    if model_name == "whyxrayclip":
        model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whyxrayclip")
        model.to(device)
        model.eval()
    
    elif model_name == "whylesionclip":
        model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whylesionclip")
        model.to(device)
        model.eval()
    
    elif model_name == "whyxrayclip_PubMed":
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="./data/model_weights/whyxrayclip_PubMed.pt")
        model.to(device)
        model.eval()
    
    elif model_name == "whylesionclip_PubMed":
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="./data/model_weights/whylesionclip_PubMed.pt")
        model.to(device)
        model.eval()

    elif model_name == "openclip":
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
        model.to(device)
        model.eval()

    elif model_name == "vit_random":
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=False)
        model.to(device)
        model.eval()

    elif model_name == "convnext_random":
        model, _, preprocess = open_clip.create_model_and_transforms("convnext_large_d_320", pretrained=False)
        model.to(device)
        model.eval()

    elif model_name == "biomedclip":
        model, preprocess = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model.to(device)
        model.eval()
        
    elif model_name == "pmc":
        model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:ryanyip7777/pmc_vit_l_14')
        model.to(device)
        model.eval()
    
    if model_name == "clip":
        model, preprocess = clip.load("ViT-L/14", device=device)
    
    elif model_name == "pixel":
        model = PixelFeatureExtractor()
        preprocess = model.preprocess
    
    elif model_name == "medclip":
        model = MedCLIPFeatureExtractor()
        preprocess = model.preprocess
    
    elif model_name == "pubmedclip":
        model = PubMedFeatureExtractor()
        preprocess = model.preprocess
    
    elif model_name == "DenseNetXray":
        model = DenseNetXrayFeatureExtractor()
        preprocess = model.preprocess
    
    elif model_name == "DenseNetSkin":
        model = DenseNetSkinFeatureExtractor()
        preprocess = model.preprocess
    
    return model, preprocess


def get_embeddings(data_loader, model, save_dir):
    i = 0
    for sample in tqdm(data_loader):
        image_path, image = sample
        with torch.no_grad():
            image_features = model.encode_image(image.to(device))
        i += 1
        features = {image_path:embedding for image_path, embedding in zip(image_path, image_features)}
        torch.save(features, f'{save_dir}/{i}.pt')


def open_image(image_path):
    # Open the main image in RGB mode
    return Image.open(image_path).convert("RGB")


def extract_helper(t_list, key, batch_size, dataset_dir, output_path, model, preprocess, image_dir):
    print(f"Currently running {key}")
    if not os.path.exists(output_path + key):
        os.makedirs(output_path + key)
    batch_count = 0

    # batch extract
    final_img = []

    f_t_list = [image_dir + ttt for ttt in t_list]
    num_batch = int(len(f_t_list) / batch_size) + 1

    for b_idx in tqdm(range(num_batch)):
        if b_idx == num_batch - 1:
            # Last batch!
            start_idx = b_idx * batch_size
            if start_idx == len(f_t_list):
                continue
            else: 
                try: images = torch.stack([preprocess(open_image(img)) for img in f_t_list[start_idx:]]).to(device)
                except: images = [preprocess(open_image(img)) for img in f_t_list[start_idx:]]
        else:
            start_idx = b_idx * batch_size
            end_idx = (b_idx + 1) * batch_size

            try: images = torch.stack([preprocess(open_image(img)) for img in f_t_list[start_idx:end_idx]]).to(device)
            except: images = [preprocess(open_image(img)) for img in f_t_list[start_idx:end_idx]]

        with torch.no_grad():
            image_features = model.encode_image(images)
            try: image_features = image_features.cpu().numpy()
            except: image_features = np.array(image_features)

        final_img.extend(image_features)

        batch_count += 1
        if batch_count == 40:
            # Save this!
            final_img = np.array(final_img)
            output_file_path = output_path + key + "/" + f"{b_idx}.pt"
            torch.save(final_img, output_file_path)
            # Clear
            batch_count = 0
            final_img = []
        
    # After finish check if we have remaining things in final_img
    if len(final_img) != 0:
        # Save them!
        final_img = np.array(final_img)
        output_file_path = output_path + key + "/" + f"{num_batch}.pt"
        torch.save(final_img, output_file_path)


def extract_vision(model, preprocess, split_dict, batch_size, img_dir_path, output_path, data_name, model_name, image_dir):
    label_list = list(split_dict.keys())
    label2index = {label_list[i]:i for i in range(len(label_list))}

    torch.save(label2index, f"data/features/{model_name}/{data_name}_label.pt")
    
    for ll in label_list:
        yes_list = split_dict[ll]
        extract_helper(yes_list, ll, batch_size, img_dir_path, output_path, model, preprocess, image_dir)


def extracted_features_dataset(dataset_dir, image_dir, model_name, dataset_name):
    model, preprocess = load_model(model_name)

    if not os.path.exists(f"./data/features/{model_name}/{dataset_name}"):
        os.makedirs(f"./data/features/{model_name}/{dataset_name}")

    split_path = dataset_dir + dataset_name + "/splits/"
    for split in ["train", "val", "test"]:
        with open(split_path + f"class2images_{split}.p", 'rb') as f:
            split_dict = pickle.load(f)
            output_path = f"./data/features/{model_name}/{dataset_name}/{split}/"

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            extract_vision(model, preprocess, split_dict, 64, dataset_dir, output_path, dataset_name, model_name, image_dir)

    print(f"Datasets {dataset_name} finished extracting")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="NIH-sex")
    parser.add_argument("--model_name", type=str, default="whyxrayclip")
    parser.add_argument("--dataset_dir", type=str, default="./data/datasets/")
    parser.add_argument("--image_dir", type=str, default="./data/datasets/") # This is the path to where the images are stored

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    dataset_dir = args.dataset_dir
    image_dir = args.image_dir

    extracted_features_dataset(dataset_dir, image_dir, model_name, dataset_name)