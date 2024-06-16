import torch
import torch.nn as nn

class LogisticRegressionT(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionT, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class MultiLabelModel(nn.Module):
    def __init__(self, model, num_classes):
        super(MultiLabelModel, self).__init__()
        self.num_classes = num_classes
        self.vision_encoder = model.visual
        self.linear = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.vision_encoder(x)
        x = self.linear(x)
        return x


class MultiClassLogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes, prior, apply_prior=False):
        super(MultiClassLogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)
        self.prior = prior.cuda() # prior is a weight matrix has the same shape as the weight matrix of the linear layer
        self.apply_prior = apply_prior
    
    def forward(self, x):
        return self.linear(x)


class PosthocHybridCBM(nn.Module):
    def __init__(self, n_concepts, n_classes, n_image_features, apply_prior=False):
        """
        PosthocCBM Hybrid Layer. 
        Takes an embedding as the input, outputs class-level predictions.
        Uses both the embedding and the concept predictions.
        Args:
            bottleneck (PosthocLinearCBM): [description]
        """
        super(PosthocHybridCBM, self).__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.n_image_features = n_image_features
        self.apply_prior = apply_prior

        self.bottleneck_classifier = nn.Linear(self.n_concepts, self.n_classes)
        self.residual_classifier = nn.Linear(self.n_image_features, self.n_classes)

    def forward(self, features):
        image_features = features[:, :self.n_image_features]
        concept_features = features[:, self.n_image_features:]

        out = self.bottleneck_classifier(concept_features) + self.residual_classifier(image_features)
        return out


class DenseNetE2E(nn.Module):
    def __init__(self, denset_model, num_classes):
        super(DenseNetE2E, self).__init__()
        self.denset_model = denset_model
        self.linear_layer = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.denset_model.features2(x)
        x = self.linear_layer(x)
        return x


class ViTE2E(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(ViTE2E, self).__init__()
        self.vision_encoder = clip_model.visual
        self.linear_layer = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.vision_encoder(x)
        x = self.linear_layer(x)
        return x


class CLIPBinary(nn.Module):
    def __init__(self, clip_model):
        super(CLIPBinary, self).__init__()
        self.clip_model = clip_model
        self.linear = nn.Linear(1536, 1)

    def forward(self, images, texts):
        text_features = self.clip_model.encode_text(texts)
        image_features = self.clip_model.encode_image(images)
        x = torch.cat((text_features, image_features), dim=1)
        x = torch.sigmoid(self.linear(x))
        return x