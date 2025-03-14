import torch

def accuracy(pred, y):
    # accuracy = 1 - (norm(y - pred) / norm(y))
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")


def r2(pred, y):
    # R square (coefficient of determination) = 1 - (sum((y - pred)^2) / sum((y - mean(y))^2))
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)


def explained_variance(pred, y):
    # explained variance = 1 - var(y - pred) / var(y)
    return 1 - torch.var(y - pred) / torch.var(y)