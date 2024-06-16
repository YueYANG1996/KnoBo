import json
import copy
import torch
import pandas as pd
from argparse import ArgumentParser

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from utils import *
from models import MultiClassLogisticRegression, PosthocHybridCBM

import random
random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device being used:", device)

def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs):
    best_val_acc = -float("inf")
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.long())

            if model.apply_prior != False:
                prior_loss = compute_prior_loss(model)
                loss += 1.0 * prior_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # evaluate the model
        val_acc = evaluate_model(model, val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader)}, Val Acc: {val_acc}")
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
    
    return best_model


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = 100 * correct / total
    
    return accuracy


def get_results(args, label2index, classifier_list, df_train_log, y_train, df_val_log, y_val, df_ood_log, y_ood, batch_size, learning_rate, num_epochs):
    # Convert features and labels to PyTorch tensors
    X_train_torch = torch.tensor(df_train_log.values).float().to(device)
    y_train_torch = torch.tensor(y_train).to(device)
    X_val_torch = torch.tensor(df_val_log.values).float().to(device)
    y_val_torch = torch.tensor(y_val).to(device)
    X_ood_torch = torch.tensor(df_ood_log.values).float().to(device)
    y_ood_torch = torch.tensor(y_ood).to(device)

    # Create DataLoader instances
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    val_dataset = TensorDataset(X_val_torch, y_val_torch)
    ood_dataset = TensorDataset(X_ood_torch, y_ood_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(torch.unique(y_train_torch))  # Assuming y_train contains all classes
    class_names = list(label2index.keys())
    concepts = list(classifier_list.keys())

    # Get the prior matrix
    prior = get_prior_matrix(args.modality, class_names, concepts)

    # Define the logistic regression model
    if args.mode == "pcbm":
        model = PosthocHybridCBM(n_concepts=len(concepts), 
                                 n_classes=num_classes, 
                                 n_image_features=X_train_torch.shape[1] - len(concepts))
    else:
        model = MultiClassLogisticRegression(num_features=X_train_torch.shape[1], 
                                             num_classes=num_classes, 
                                             prior=prior,
                                             apply_prior=args.add_prior)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    best_model = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs)

    # Evaluate the model
    val_acc = evaluate_model(best_model, val_loader)
    ood_acc = evaluate_model(best_model, ood_loader)

    average_acc = round((val_acc + ood_acc) / 2, 2)
    gap = round(abs(val_acc - ood_acc), 2)
    
    return val_acc, ood_acc, gap, average_acc, best_model


def run_classification(args, dataset_name, classifier_list, clip_model, tokenizer, preprocess, random_seed=42):
    # Load the features
    label2index = torch.load(f"./data/features/{args.model_name}/{dataset_name}_label.pt")
    X_train, y_train, X_val, y_val, X_ood, y_ood = load_features(f"./data/features/{args.model_name}/{dataset_name}", label2index, args.shots, args.normalize, random_seed)

    if args.mode == "binary":
        df_train_log, df_val_log, df_ood_log = binary_features(X_train, X_val, X_ood, classifier_list)
    elif args.mode == "linear_probe":
        df_train_log, df_val_log, df_ood_log = linear_features(X_train, X_val, X_ood, args.number_of_features)
    elif args.mode == "dot_product":
        df_train_log, df_val_log, df_ood_log = dot_product_features(X_train, X_val, X_ood, classifier_list, clip_model, tokenizer)
    elif args.mode == "pcbm":
        df_train_log, df_val_log, df_ood_log = pcbm_features(X_train, X_val, X_ood, classifier_list, clip_model, tokenizer, preprocess, args.number_of_features)

    print("Train size: ", df_train_log.shape, "Test size: ", df_val_log.shape, "OOD size: ", df_ood_log.shape)

    val_acc, ood_acc, gap, average_acc, best_model = get_results(args, label2index, classifier_list, df_train_log, y_train, df_val_log, y_val, df_ood_log, y_ood, batch_size=64, learning_rate=0.001, num_epochs=200)

    print(f"Dataset: {dataset_name}, Mode: {args.mode}", f"Shots: {args.shots}", f"Model: {args.model_name}")
    print(f"Ind Acc: {val_acc}, OOD Acc: {ood_acc}, Gap: {gap}, Average: {average_acc}")
    number_of_features_actual = df_train_log.shape[1]

    return val_acc, ood_acc, gap, average_acc, number_of_features_actual


def run_all_datasets(args, dataset_lists, classifier_list, clip_model, tokenizer, preprocess):
    results_dict = {}
    for dataset_name in dataset_lists:
        ind_acc, out_acc, gap, avg, number_of_features_actual = run_classification(args, dataset_name, classifier_list, clip_model, tokenizer, preprocess)
        results_dict[dataset_name] = {"ind_acc": ind_acc, "out_acc": out_acc, "gap": gap, "avg": avg}

    # reshape to one row
    csv_df = pd.DataFrame.from_dict(results_dict).T
    # Reshape the data
    reshaped_df = pd.DataFrame(csv_df.values.flatten()).T

    # Create new column names
    new_columns = [f"{row_label}_{col_label}" for row_label in csv_df.index for col_label in csv_df.columns]

    # Assign new column names to reshaped dataframe
    reshaped_df.columns = new_columns
    
    # save as csv use all arguments as file name
    file_name = f"./data/results/{args.modality}_{args.mode}_{args.model_name}_{args.bottleneck}_{args.shots}_{number_of_features_actual}_{args.save_suffix}.csv"

    if args.add_prior: file_name = file_name.replace(".csv", "_prior.csv")
    
    # creat folder if not exist
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    reshaped_df.to_csv(file_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="binary", help="binary, linear_probe, dot_product")
    parser.add_argument("--bottleneck", type=str, default="PubMed", help="PubMed, prompt, Textbooks")
    parser.add_argument("--shots", type=str, default="all", help="all, 1, 2, 4, 8, 16, 32, 64")
    parser.add_argument("--model_name", type=str, default="whyxrayclip", help="whyxrayclip, whylesionclip")
    parser.add_argument("--acc_threshold", type=float, default=0, help="accuracy threshold for loading classifiers")
    parser.add_argument("--number_of_features", type=int, default=768, help="number of features/concepts")
    parser.add_argument("--normalize", type=bool, default=True, help="normalize the features")
    parser.add_argument("--modality", type=str, default="xray", help="xray, natural, skin")
    parser.add_argument("--input_dim", type=int, default=768, help="input dimension for the binary classifiers")
    parser.add_argument("--add_prior", type=bool, default=False, help="add prior to the model")
    parser.add_argument("--save_suffix", type=str, default="", help="add suffix to the save folder")

    args = parser.parse_args()

    # Load clip model
    clip_model, tokenizer, preprocess = load_clip_model(args.model_name)

    # Load classifiers
    classifier_list = load_classifier_list(args)
    binary_accuracies = [classifier_list[k][1] for k in classifier_list.keys()]
    print(f"Number of classifiers: {len(classifier_list)}, Mean Acc: {round(sum(binary_accuracies) / len(binary_accuracies), 5)}")

    # Load datasets
    if args.modality == "xray":
        dataset_lists = ["NIH-sex", "NIH-age", "NIH-pos", "CheXpert-race", "NIH-CheXpert", "pneumonia", "COVID-QU", "NIH-CXR", "open-i", "vindr-cxr"]
    elif args.modality == "skin":
        dataset_lists = ["ISIC-sex", "ISIC-age", "ISIC-site", "ISIC-color", "ISIC-hospital", "HAM10000", "BCN20000", "PAD-UFES-20", "Melanoma", "UWaterloo"]
        
    run_all_datasets(args, dataset_lists, classifier_list, clip_model, tokenizer, preprocess)