import os
import json
import random
import wandb
import open_clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
from utils import load_clip_model
random.seed(0)

# This script will fine-tune clip with the knowledge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# dataloader
class FinetuneDataset(Dataset):
    def __init__(self, data, image_dir, preprocess, tokenizer):
        self.data = data
        self.preprocess = preprocess
        self.image_paths = list(set([d[0] for d in data]))
        self.texts = list(set([d[1] for d in data]))

        print("Preprocessing images ...") # you need a lot of memory for this
        self.image_path2image = {image_path: preprocess(Image.open(image_dir + image_path)) for image_path in tqdm(self.image_paths)}

        print("Tokenizing texts ...")
        self.text2token = {text: tokenizer(text) for text in tqdm(self.texts)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, text, label = self.data[idx]
        image = self.image_path2image[image_path]
        text = self.text2token[text]
        return image, text, label


def get_label_for_concept(args, features, metadata, annotations, concept):
    positive = annotations[concept]["positive"]
    negative = annotations[concept]["negative"]
    
    positive_images = []
    negative_images = []

    if args.modality == "xray":
        for report_id in positive:
            images = metadata[report_id]["images"]
            for image, image_type in images:
                if image_type in ["AP", "PA"] and image in features:
                    positive_images.append(image)
        
        for report_id in negative:
            images = metadata[report_id]["images"]
            for image, image_type in images:
                if image_type in ["AP", "PA"] and image in features:
                    negative_images.append(image)
    
    elif args.modality == "skin":
        for report_id in positive:
            images = metadata[report_id]["images"]
            for image in images:
                if image in features:
                    positive_images.append(image)
        
        for report_id in negative:
            images = metadata[report_id]["images"]
            for image in images:
                if image in features:
                    negative_images.append(image)

    random.seed(0)
    random.shuffle(positive_images)
    random.shuffle(negative_images)

    # equally add positive and negative examples up to max_examples
    if len(positive_images) > len(negative_images):
        negative_images_selected = negative_images[:min(len(negative_images), args.max_examples//2)]
        positive_images_selected = positive_images[:args.max_examples - len(negative_images_selected)]
    else:
        positive_images_selected = positive_images[:min(len(positive_images), args.max_examples//2)]
        negative_images_selected = negative_images[:args.max_examples - len(positive_images_selected)]
    
    val_len = min(int(0.1*min(len(positive_images_selected), len(negative_images_selected))), 50)

    if val_len < 10:
        print(f"Test length too small for {concept}. Skipping ...")
        return False

    positive_train, positive_val = train_test_split(positive_images_selected, test_size=val_len, random_state=0)
    negative_train, negative_val = train_test_split(negative_images_selected, test_size=val_len, random_state=0)

    positive_train = positive_train[:int(args.train_samples*0.5)]
    negative_train = negative_train[:args.train_samples - len(positive_train)]
    
    # downsample to keep the training data balanced
    random.seed(0)
    if len(positive_train) > len(negative_train): positive_train = random.sample(positive_train, len(negative_train))
    else: negative_train = random.sample(negative_train, len(positive_train))

    data = {"positive": {"train": positive_train, "val": positive_val}, "negative": {"train": negative_train, "val": negative_val}}

    print(f"Question: {concept}, Positive: {len(positive_train)}, Negative: {len(negative_train)}")
    return data


def get_training_data(args, features, metadata, annotations):
    with open(f"./data/bottlenecks/{args.modality}_{args.bottleneck}.txt", "r") as f:
        concepts = f.readlines()

    concepts = [concept.strip() for concept in concepts]
    concept2annotations = {concept: get_label_for_concept(args, features, metadata, annotations, concept) for concept in concepts}

    train_examples = []
    val_examples = []
    label2idx = {"positive": 1, "negative": 0}
    
    for concept, data in concept2annotations.items():
        if data:
            for label, split in data.items():
                for image in split["train"]:
                    train_examples.append((image, concept, label2idx[label]))
                for image in split["val"]:
                    val_examples.append((image, concept, label2idx[label]))
    
    return train_examples, val_examples


def contrastive_loss(similarities, labels, margin=0.6):
    """Compute the contrastive loss based on cosine similarities."""

    loss_similar = labels * (margin - similarities).clamp(min=0)
    loss_dissimilar = (1 - labels) * similarities

    loss = loss_similar + loss_dissimilar

    return loss.mean()


def finetune_clip(args, features, metadata, annotations):
    wandb.init(project="finetune_clip", 
               name=f"{args.clip_model_name}_{args.bottleneck}_{args.batch_size}_{args.epochs}",
               config={
                "bottleneck": args.bottleneck,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "clip_model_name": args.clip_model_name}
               )

    # get the training data
    train_data, val_data = get_training_data(args, features, metadata, annotations)

    print("Number of training examples:", len(train_data))
    print("Number of validation examples:", len(val_data))

    # get the model
    clip_model, tokenizer, preprocess = load_clip_model(args.clip_model_name)
    clip_model.to(device)

    # get the dataloader
    train_data = FinetuneDataset(train_data, args.image_dir, preprocess, tokenizer)
    val_data = FinetuneDataset(val_data, args.image_dir, preprocess, tokenizer)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # the label of each example is binary: 0 or 1, models' outputs are cosine similarities
    optimizer = optim.Adam(clip_model.parameters(), lr=args.learning_rate, weight_decay=1e-6)

    best_val_loss = float("inf")

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(args.epochs):
        clip_model.train()
        for i, (images, texts, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            text_features = clip_model.encode_text(texts.squeeze().to(device))
            image_features = clip_model.encode_image(images.to(device))

            labels = labels.float().to(device)

            # Normalize features to prevent in-place modification issues
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute the dot product between text and image features
            similarity_matrix = image_features @ text_features.t()
            logits = torch.diag(similarity_matrix)  # Get the diagonal elements of the similarity matrix

            loss = contrastive_loss(logits, labels)

            loss.backward()
            optimizer.step()

            # Log training loss at each iteration
            wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": epoch * len(train_loader) + i})

        clip_model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, texts, labels in val_loader:
                text_features = clip_model.encode_text(texts.squeeze().to(device))
                text_features /= text_features.norm(dim=-1, keepdim=True)

                image_features = clip_model.encode_image(images.to(device))
                image_features /= image_features.norm(dim=-1, keepdim=True)

                labels = labels.float().to(device)

                similarity_matrix = image_features @ text_features.t()
                logits = torch.diag(similarity_matrix)

                loss = contrastive_loss(logits, labels)
                val_loss += loss.item()

        # Log validation loss and accuracy at the end of each epoch
        wandb.log({"val_loss": val_loss / len(val_loader), "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(clip_model.state_dict(), f"./data/model_weights/{clip_model_name}_{bottleneck}.pt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--modality", type=str, default="xray")
    parser.add_argument("--bottleneck", type=str, default="PubMed")
    parser.add_argument("--image_dir", type=str, default="./data/datasets/MIMIC-CXR/images/")
    parser.add_argument("--clip_model_name", type=str, default="whyxrayclip")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_examples", type=int, default=10000)
    parser.add_argument("--train_samples", type=int, default=2000)
    
    args = parser.parse_args()

    print("Loading features/metadata/annotations ...")
    if args.modality == "xray":
        features = torch.load(f'./data/datasets/MIMIC-CXR/MIMIC-CXR_whyxrayclip.pt')
        metadata = json.load(open('./data/datasets/MIMIC-CXR/MIMIC-CXR_metadata.json', 'r'))
        annotations = json.load(open('./data/datasets/MIMIC-CXR/MIMIC-CXR_concept_annotations.json', 'r'))

    elif args.modality == "skin":
        features = torch.load(f'./data/datasets/ISIC/ISIC_whylesionclip.pt')
        metadata = json.load(open('./data/datasets/ISIC/ISIC_metadata.json', 'r'))
        annotations = json.load(open('./data/datasets/ISIC/ISIC_concept_annotations.json', 'r'))

    finetune_clip(args, features, metadata, annotations)