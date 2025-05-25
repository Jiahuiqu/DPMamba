import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class ClipLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.08))

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features):
        image_features = image_features / torch.clamp(image_features.norm(dim=1, keepdim=True), min=1e-8)
        text_features = text_features / torch.clamp(text_features.norm(dim=1, keepdim=True), min=1e-8)

        logit_scale = self.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, cal_logits=False):
        device = image_features.device

        logits_per_image, logits_per_text = self.get_logits(image_features=image_features, text_features=text_features)


        if cal_logits:
            return (logits_per_image, logits_per_text)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (F.cross_entropy(logits_per_image, labels) +
                      F.cross_entropy(logits_per_text, labels)) / 2

        if torch.isnan(total_loss) or torch.isinf(total_loss):
             raise("Invalid loss value detected!")

        return total_loss

def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1., gamma=1.):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        if y_s.ndim == 4:
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        y_s = y_s.softmax(dim=1)
        y_t = y_t.softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        return inter_loss