import torch.nn.functional as F
import torch

# def l1_l2_loss(pred, true, l1_weight, scores_dict):
#     """
#     Regularized MSE loss; l2 loss with l1 loss too.

#     Parameters
#     ----------
#     pred: torch.floatTensor
#         The model predictions
#     true: torch.floatTensor
#         The true values
#     l1_weight: int
#         The value by which to weight the l1 loss
#     scores_dict: defaultdict(list)
#         A dict to which scores can be appended.

#     Returns
#     ----------
#     loss: the regularized mse loss
#     """
#     loss = F.mse_loss(pred, true)

#     scores_dict['l2'].append(loss.item())

#     if l1_weight > 0:
#         l1 = F.l1_loss(pred, true)
#         loss += l1
#         scores_dict['l1'].append(l1.item())
#     scores_dict['loss'].append(loss.item())

#     return loss, scores_dict

def l1_l2_loss(pred, true, l1_weight, running_scores):
    loss = F.mse_loss(pred, true)
    if l1_weight > 0:
        l1_loss = F.l1_loss(pred, true)
        loss += l1_weight * l1_loss

    running_scores['loss'].append(loss.item())
    return loss, running_scores


def huber_loss(pred, true, delta=1.0, running_scores=None):
    """
    Huber loss - a loss function that's less sensitive to outliers in the data.
    L2 for small residuals, L1 for large residuals.
    """
    loss = torch.nn.functional.huber_loss(pred, true, delta=delta)
    
    if running_scores is not None:
        running_scores['loss'].append(loss.item())
    
    return loss, running_scores