import torch


def accuracy(outputs, labels):
    with torch.no_grad():
        predict_label = torch.argmax(outputs, dim=1)
        equal_num = torch.sum(predict_label == labels).item()
        total_num = torch.numel(labels)
        acc = equal_num / total_num
    return acc


def fpr_fnr(all_score, all_label):
    with torch.no_grad():
        same = torch.eq(all_label[0], all_label[1]).to(dtype=torch.int)
        all_score, sorted_indices = torch.sort(all_score)
        same = same[sorted_indices]
        diff = (same == 0).to(dtype=torch.int)
        if torch.sum(diff) == 0:
            fpr = torch.zeros_like(diff)
        else:
            fpr = 1 - (torch.cumsum(diff, dim=0) - 1) / torch.sum(diff)
        if torch.sum(same) == 0:
            fnr = torch.zeros_like(same)
        else:
            fnr = torch.cumsum(same, dim=0) / torch.sum(same)
        return fpr, fnr, all_score


def eer_threshold(fpr, fnr, all_score):
    with torch.no_grad():
        diff = fpr - fnr
        if torch.nonzero(diff == 0).numel() > 0:
            num = torch.nonzero(diff == 0).numel()
            x = torch.nonzero(diff == 0)[num // 2]
            eer = fpr[x].item()
            threshold = all_score[x].item()
        else:
            x1 = torch.nonzero(diff > 0)
            x2 = torch.nonzero(diff < 0)
            if x1.numel() > 0 and x2.numel() > 0:
                x1 = x1.max()
                x2 = x2.min()
                threshold = (all_score[x2] - all_score[x1]).item() / 2
                eer = (fpr[x1] + fnr[x1] + fpr[x2] + fnr[x2]).item() / 4
            elif x1.numel() == 0:
                threshold = all_score[0].item()
                eer = 0
            elif x2.numel() == 0:
                threshold = all_score[-1].item()
                eer = 0
    return eer, threshold


# def fpr_fnr(score, one_hot):
#     sorted_indices = torch.argsort(score)
#     one_hot = one_hot[sorted_indices]
#     one_hot_re = (one_hot == 0)
#     fpr = 1 - torch.cumsum(one_hot_re, dim=0) / torch.sum(one_hot_re)
#     fnr = torch.cumsum(one_hot, dim=0) / torch.sum(one_hot)
#     return fpr, fnr


# def eer_threshold(score, one_hot):
#     sorted_indices = torch.argsort(score)
#     one_hot = one_hot[sorted_indices]
#     one_hot_re = (one_hot == 0)
#     fpr = 1 - torch.cumsum(one_hot_re, dim=0) / torch.sum(one_hot_re)
#     fnr = torch.cumsum(one_hot, dim=0) / torch.sum(one_hot)
#     diff = fpr - fnr
#     x1 = torch.nonzero(diff > 0).max()
#     x2 = torch.nonzero(diff < 0).min()
#     threshold = (fpr[x1] * fnr[x2] - fpr[x2] * fnr[x1]) / (fpr[x1] - fpr[x2] - fnr[x1] + fnr[x2])
#     eer = (fpr[x1] + fnr[x1]) / 2
#     return eer, threshold
