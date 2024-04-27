import torch


def accuracy(outputs, labels):
    with torch.no_grad():
        predict_label = torch.argmax(outputs, dim=1)
        equal_num = torch.sum(predict_label == labels).item()
        total_num = torch.numel(labels)
        acc = equal_num / total_num
    return acc


def cal_fpr_fnr(all_score, all_label):
    with torch.no_grad():
        all_score, sorted_indices = torch.sort(all_score)
        all_label = all_label[sorted_indices]
        same = torch.where(all_label, 1, 0)
        diff = torch.where(all_label, 0, 1)
        if torch.sum(diff) == 0:
            fpr = torch.zeros_like(diff)
        else:
            fpr = 1 - (torch.cumsum(diff, dim=0) - 1) / torch.sum(diff)
        if torch.sum(same) == 0:
            fnr = torch.zeros_like(same)
        else:
            fnr = torch.cumsum(same, dim=0) / torch.sum(same)
        return fpr, fnr, all_score


def cal_eer_threshold(fpr, fnr, all_score):
    with torch.no_grad():
        diff = fpr - fnr
        if torch.nonzero(diff == 0).numel() > 0:
            num = torch.nonzero(diff == 0).numel()
            x = torch.nonzero(diff == 0)[num // 2]
            eer = fpr[x].item()
            eer_threshold = all_score[x].item()
        else:
            x1 = torch.nonzero(diff > 0)
            x2 = torch.nonzero(diff < 0)
            if x1.numel() > 0 and x2.numel() > 0:
                x1 = x1.max()
                x2 = x2.min()
                eer_threshold = (all_score[x1] + all_score[x2]).item() / 2
                eer = (fpr[x1] + fnr[x1] + fpr[x2] + fnr[x2]).item() / 4
            elif x1.numel() == 0:
                eer_threshold = all_score[0].item()
                eer = 0
            elif x2.numel() == 0:
                eer_threshold = all_score[-1].item()
                eer = 0
    return eer, eer_threshold


def cal_min_dcf_threshold(fpr, fnr, all_score, p_target=1e-2, c_fa=1, c_miss=1):
    dcf = c_miss * p_target * fpr + c_fa * (1 - p_target) * fnr
    dcf, sorted_indices = torch.sort(dcf)
    all_score = all_score[sorted_indices]
    min_dcf = dcf[0].item()
    min_dcf_threshold = all_score[0].item()
    return min_dcf, min_dcf_threshold
