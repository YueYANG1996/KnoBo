import os
import json
import torch
import pickle
import open_clip
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

import random
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_classifier_list(args):
    with open(f"./data/bottlenecks/{args.modality}_{args.bottleneck}.txt", "r") as f:
        concepts = f.read().strip().split("\n")
    
    model_path = f"./data/grounding_functions/{args.modality}"
    classifier_list = {}
    if args.bottleneck == "random":
        for c in concepts:
            classifier_list[c] = [LogisticRegression(max_iter=1000), 0.5]
    else:
        for c in concepts:
            model_full_path = f"{model_path}/{c}/{c}.p"
            # check existence of the model
            if not os.path.exists(model_full_path):
                print(f"Model for {c} does not exist")
                continue
            curr_model = pickle.load(open(model_full_path, "rb"))

            with open(f"{model_path}/{c}/{c}_results.txt", "r") as f:
                results = f.read().strip().split(",")
                val_acc = float(results[-1].strip())

            if val_acc >= args.acc_threshold:
                classifier_list[c] = (curr_model, val_acc)
    
    # sort the classifiers based on the accuracy
    classifier_list = {k: v for k, v in sorted(classifier_list.items(), key=lambda item: item[1][1], reverse=True)}
    if args.number_of_features < len(classifier_list):
        # pick top number_of_features classifiers with highest accuracy
        classifier_list = {k: classifier_list[k] for k in list(classifier_list.keys())[:args.number_of_features]}

    return classifier_list


def load_clip_model(model_name):
    if model_name == "whyxrayclip":
        clip_model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whyxrayclip")
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

    elif model_name == "whylesionclip":
        clip_model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whylesionclip")
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

    elif model_name == "openclip":
        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

    elif model_name == "openclip_random":
        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=False)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

    elif model_name == "convnext_random":
        clip_model, _, preprocess = open_clip.create_model_and_transforms("convnext_large_d_320", pretrained=False)
        tokenizer = open_clip.get_tokenizer('convnext_large_d_320')

    else:
        clip_model = None
        tokenizer = None
        preprocess = None
    
    return clip_model, tokenizer, preprocess


def linear_features(X_train, X_val, ood_features, number_of_features):
    selected_indices = random.sample(range(0, X_train.shape[1]), min(number_of_features, X_train.shape[1]))
    df_train_log = pd.DataFrame(X_train[:, selected_indices])
    df_val_log = pd.DataFrame(X_val[:, selected_indices])
    df_ood_log = pd.DataFrame(ood_features[:, selected_indices])

    return df_train_log, df_val_log, df_ood_log


def binary_features(X_train_features, X_val_features, ood_features, classifier_list):
    binary_logits_train = {}
    binary_logits_val = {}
    binary_logits_ood = {}

    for kk in classifier_list.keys():
        lr_model = classifier_list[kk][0]

        binary_logits_train[kk] = lr_model.predict_proba(X_train_features)[:, 1]
        binary_logits_val[kk] = lr_model.predict_proba(X_val_features)[:, 1]
        binary_logits_ood[kk] = lr_model.predict_proba(ood_features)[:, 1]

    df_train_log = pd.DataFrame.from_dict(binary_logits_train)
    df_val_log = pd.DataFrame.from_dict(binary_logits_val)
    df_ood_log = pd.DataFrame.from_dict(binary_logits_ood)
    
    return df_train_log, df_val_log, df_ood_log


def dot_product_features(X_train, X_val, ood_features, classifier_list, clip_model, tokenizer):
    prompt_list = list(classifier_list.keys())
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_model.encode_text(tokenizer(prompt_list))
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_features = text_features.numpy()
    product_train = X_train @ text_features.T
    product_val = X_val @ text_features.T
    product_ood = ood_features @ text_features.T

    df_train_log = pd.DataFrame(product_train)
    df_val_log = pd.DataFrame(product_val)
    df_ood_log = pd.DataFrame(product_ood)

    return df_train_log, df_val_log, df_ood_log


def pcbm_features(X_train, X_val, ood_features, classifier_list, clip_model, tokenizer, preprocess, number_of_features):
    # ensemble linear features and dot product features
    df_train_log_lin, df_val_log_lin, df_ood_log_lin = linear_features(X_train, X_val, ood_features, 768)
    df_train_log_dot, df_val_log_dot, df_ood_log_dot = dot_product_features(X_train, X_val, ood_features, classifier_list, clip_model, tokenizer)

    df_train_log = pd.concat([df_train_log_lin, df_train_log_dot], axis=1)
    df_val_log = pd.concat([df_val_log_lin, df_val_log_dot], axis=1)
    df_ood_log = pd.concat([df_ood_log_lin, df_ood_log_dot], axis=1)

    return df_train_log, df_val_log, df_ood_log    


def load_features(feature_path, label2index, shots, normalize, random_seed):
    train_path = f"{feature_path}/train"
    val_path = f"{feature_path}/val"
    ood_path = f"{feature_path}/test"

    tmp_train_list = []
    tmp_val_list = []
    tmp_train_label = []
    tmp_val_label = []
    label_list = list(label2index.keys())
    ood_list = []
    ood_label = []

    for ll in tqdm(label_list):
        # try:
        train_tmp = []
        val_tmp = []
        ood_tmp = []

        # Get train val val
        train_path_list = [f"{train_path}/{ll}/{yp}" for yp in os.listdir(f"{train_path}/{ll}")]
        val_path_list = [f"{val_path}/{ll}/{yp}" for yp in os.listdir(f"{val_path}/{ll}")]
        ood_path_list = [f"{ood_path}/{ll}/{yp}" for yp in os.listdir(f"{ood_path}/{ll}")]

        for yp in train_path_list:
            train_tmp.extend(torch.load(yp))

        tmp_train_label.extend([label2index[ll]] * len(train_tmp))

        for tp in val_path_list:
            val_tmp.extend(torch.load(tp))

        tmp_val_label.extend([label2index[ll]] * len(val_tmp))

        for opp in ood_path_list:
            ood_tmp.extend(torch.load(opp))

        ood_label.extend([label2index[ll]] * len(ood_tmp))
        tmp_train_list.extend(train_tmp)
        tmp_val_list.extend(val_tmp)
        ood_list.extend(ood_tmp)

    df_train_tmp = pd.DataFrame(tmp_train_list)
    df_train_tmp["labels"] = tmp_train_label

    df_val_tmp = pd.DataFrame(tmp_val_list)
    df_val_tmp["labels"] = tmp_val_label

    df_ood = pd.DataFrame(ood_list)
    df_ood["labels"] = ood_label

    df_train= df_train_tmp.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_val = df_val_tmp.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    if shots != "all":
        # random sample number of shots for each class
        df_train = df_train.groupby('labels').sample(n=int(shots), random_state=random_seed).reset_index(drop=True)
        # shuffle the train set with fixed seed
        df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    X_train = torch.tensor(df_train[list(df_train.columns)[:-1]].values).float()
    y_train = df_train["labels"].values

    X_val = torch.tensor(df_val[list(df_val.columns)[:-1]].values).float()
    y_val = df_val["labels"].values

    ood_features = torch.tensor(df_ood[list(df_ood.columns)[:-1]].values).float()

    if normalize:
        print("Normalizing the features")
        X_train /= X_train.norm(dim=-1, keepdim=True)
        X_val /= X_val.norm(dim=-1, keepdim=True)
        ood_features /= ood_features.norm(dim=-1, keepdim=True)
    
    return X_train, y_train, X_val, y_val, ood_features, ood_label


def get_prior_matrix(modality, class_names, concepts):
    prior = torch.zeros(len(class_names), len(concepts))
    class2concept2answer = json.load(open(f"./data/bottlenecks/concept_priors/{modality}_class2concept2answer.json", "r"))

    for i, c in enumerate(class_names):
        for j, q in enumerate(concepts):
            if class2concept2answer[c][q] == "yes": prior[i, j] = 1.0
            elif class2concept2answer[c][q] == "no": prior[i, j] = -1.0
            elif class2concept2answer[c][q] == "unknown": prior[i, j] = 0.0

    return prior


def compute_prior_loss(model):
    model_weights = model.linear.weight
    prior = model.prior
    number_of_weights = model_weights.shape[0] * model_weights.shape[1]

    # apply tanh to weights to map it to [-1, 1]
    model_weights = torch.tanh(model_weights)

    # compute l1 loss between the weights and the prior
    prior_loss = torch.sum(torch.abs(model_weights - prior)) / number_of_weights

    return prior_loss


def map_weights(model, class_names, concepts):
    model.eval()
    weights = model.linear.weight
    class2concept_weights = {}
    for i, c in enumerate(class_names):
        class2concept_weights[c] = {}
        for j, q in enumerate(concepts):
            class2concept_weights[c][q] = weights[i, j].item()
    
    # sort the weights based on the absolute value
    class2concept_weights = {k: {kk: vv for kk, vv in sorted(v.items(), key=lambda item: item[1], reverse=True)} for k, v in class2concept_weights.items()}

    return class2concept_weights


def get_diversity_score(concepts):
    from sentence_transformers import SentenceTransformer, util
    sbert_model = SentenceTransformer('all-mpnet-base-v2', device = "cuda:0", cache_folder = "/nlp/data/yueyang/packages/sentence_transformer/")
    
    sentence_embeddings = sbert_model.encode(concepts, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)
    cosine_distance = 1 - cosine_scores

    # set diagonal to 1
    for i in range(len(cosine_distance)):
        cosine_distance[i][i] = 1
    
    # get the mean of each row
    mean_values = cosine_distance.mean(dim=1)

    return mean_values.mean().item()