import torch
import torch.nn as nn

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.1, beta=1.0, A=-4):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.num_classes = num_classes

        self.softmax = nn.Softmax()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        eps = torch.finfo(torch.float32).eps

        cross_entropy_loss = self.cross_entropy_loss(logits, targets)
        predictions = self.softmax(logits)
        predictions = torch.clamp(predictions, min=eps, max=1.0 - eps)
        
        label_one_hot = nn.functional.one_hot(targets, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=torch.pow(10, self.A), max=1.0)
        
        reverse_cross_entropy_loss = torch.mean(-1*torch.sum(predictions * torch.log(label_one_hot), dim=1))

        return self.alpha * cross_entropy_loss + self.beta * reverse_cross_entropy_loss

class LabelSmoothingCrossEntropyLoss(nn.Module):

    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

        self.softmax = nn.Softmax()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        targets_one_hot = nn.functional.one_hot(targets, self.num_classes).float()
        smoothed_targets = targets_one_hot * (1.0 - self.smoothing) + self.smoothing / self.num_classes
        return self.cross_entropy_loss(logits, smoothed_targets)
