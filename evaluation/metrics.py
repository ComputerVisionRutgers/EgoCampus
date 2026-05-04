import numpy as np
import torch
from tqdm import tqdm
import scipy

def auc_judd(pred, label):
    ''' Compute AUC-Judd
        pred: ndarray  [B, H, W]
        label: ndarray [B, 2]
    '''
    b,h,w = pred.shape
    assert (b, 2) == label.shape, f'{pred.shape=}, {label.shape=}'

    gt_map = np.zeros(pred.shape)
    gt = (label * (np.array(pred.shape[-2:]) - np.array([1,1]))).astype(int)
    gt_map[np.arange(gt.shape[0]), gt[:, 0], gt[:, 1]] = 1.0
    gt = gt_map

    Sth = pred[gt > 0]
    N, M = len(Sth), np.prod(pred.shape)

    if M == N:
        raise ValueError(f'{M, N, pred.shape, gt.shape, gt.sum() = }')
    
    if N == 0:
        return np.nan
    threshes = sorted(Sth, reverse=True)
    tp = np.zeros(N+2); fp = tp.copy()
    tp[0] = fp[0] = 0; tp[-1] = fp[-1] = 1

    for i, th in enumerate(threshes, start=1):
        above = np.sum(pred >= th)
        tp[i] = i / N
        fp[i] = (above - i) / (M - N)
    return np.trapezoid(tp, fp)

def cc(s_map, gt):
    if s_map.std() < 1e-7 or gt.std() < 1e-7:
        return np.nan
    a = (s_map - s_map.mean()) / (s_map.std() + 1e-7)
    b = (gt    - gt.mean())    / (gt.std()    + 1e-7)
    return (a * b).sum() / np.sqrt((a*a).sum() * (b*b).sum() + 1e-7)

def similarity(s_map, gt):
    P = s_map / (s_map.sum() + 1e-7)
    Q = gt    / (gt.sum()    + 1e-7)
    return np.sum(np.minimum(P, Q))

def kldiv(preds, label_hm):
    ''' Compute KL-Divergence
      preds: ndarray     [..., H, W]
      gazes_hm: ndarray  [..., H, W]
    '''
    h, w = preds.shape[-2:]
    n = np.array(preds.shape[:-2]).prod()
    assert (h,w) == label_hm.shape[-2:], f'{preds.shape=}, {label_hm.shape=}'
    assert n == np.array(label_hm.shape[:-2]).prod(), f'{preds.shape=}, {label_hm.shape=}'

    P = preds / (preds.sum() + 1e-7)
    Q = label_hm / (label_hm.sum()    + 1e-7)
    return np.sum(Q * np.log((Q + 1e-7) / (P + 1e-7)))

def precision_recall_f1(preds, gazes_hm, pred_threshold=0.02, gt_threshold=0.001):
    ''' Compute precision, recall, and F1
      preds: ndarray     [..., H, W]
      gazes_hm: ndarray  [..., H, W]
    '''

    pred_bin = (preds > pred_threshold).astype(np.int32)
    gt_bin = (gazes_hm > gt_threshold).astype(np.int32)

    tp = np.sum(pred_bin * gt_bin)
    fp = np.sum(pred_bin * (1 - gt_bin))
    fn = np.sum((1 - pred_bin) * gt_bin)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return precision, recall, f1

def s_auc(s_map, gt, other_gt):
    pos_samples = s_map[gt > 0]
    neg_samples = s_map[(other_gt > 0) & (gt == 0)]
    
    if len(pos_samples) == 0 or len(neg_samples) == 0:
        return np.nan
    
    labels = np.concatenate([np.ones(len(pos_samples)), np.zeros(len(neg_samples))])
    scores = np.concatenate([pos_samples, neg_samples])
    sort_indices = np.argsort(scores)[::-1]
    labels = labels[sort_indices]

    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)

    tpr = np.append([0.0], tps / tps[-1])
    fpr = np.append([0.0], fps / fps[-1])

    return np.trapezoid(tpr, fpr)

def log_likelihood(log_density_pred, labels_hm):
    l1 = np.mean(
        np.sum(log_density_pred * labels_hm) / labels_hm.shape[-1]
    )
    return (l1 + np.log(labels_hm.shape[-1] * labels_hm.shape[-2])) / np.log(2)

def information_gain(preds, labels_hm, baseline):
    ''' Compute information gain relative to a baseline
      preds: ndarray     [..., H, W]
      gazes_hm: ndarray  [..., H, W]
    '''
    assert baseline is not None

    pred_log_likelihoods = log_likelihood(preds, labels_hm)
    baseline_log_likelihoods = log_likelihood(preds, baseline)
    ig = ((pred_log_likelihoods - baseline_log_likelihoods) / np.log(2)).mean()
    return ig

def prior_penalization(labels_hm, prior):
    if labels_hm.ndim == 2:
        labels_hm = labels_hm[np.newaxis]

    if prior.ndim == 2:
        prior = prior[np.newaxis]

    labels_hm = labels_hm.reshape(labels_hm.shape[0], -1)
    prior = prior.reshape(prior.shape[0], -1)

    js_dist = scipy.spatial.distance.jensenshannon(labels_hm, prior, axis=-1)
    return js_dist ** 2

# reference: https://github.com/BolinLai/GLC/blob/main/slowfast/utils/metrics.py#L114
def adaptive_f1(preds, gazes_hm, gazes):
    thresholds = np.linspace(0, 0.02, 11) 
    all_preds = torch.zeros(size=(thresholds.shape + gazes_hm.size()), device=gazes_hm.device)
    all_labels = torch.zeros(size=(thresholds.shape + gazes_hm.size()), device=gazes_hm.device)
    binary_labels = (gazes_hm > 0.001).int()
    for i in range(thresholds.shape[0]): 
        binary_preds = (preds.squeeze(1) > thresholds[i]).int()
        all_preds[i, ...] = binary_preds
        all_labels[i, ...] = binary_labels
    tp = (all_preds * all_labels).sum(dim=(3, 4))
    fg_labels = all_labels.sum(dim=(3, 4))
    fg_preds = all_preds.sum(dim=(3, 4))

    labels_flat = gazes.view(gazes.size(0) * gazes.size(1), gazes.size(2))
    tracked_idx = torch.where(labels_flat[:, 2] != -1)[0] # what does this do?
    tp = tp.view(tp.size(0), tp.size(1)*tp.size(2)).index_select(1, tracked_idx)
    fg_labels = fg_labels.view(fg_labels.size(0), fg_labels.size(1)*fg_labels.size(2)).index_select(1, tracked_idx)
    fg_preds = fg_preds.view(fg_preds.size(0), fg_preds.size(1)*fg_preds.size(2)).index_select(1, tracked_idx)
    recall = (tp / (fg_labels + 1e-6)).mean(dim=1)
    precision = (tp / (fg_preds + 1e-6)).mean(dim=1)
    f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    max_idx = torch.argmax(f1)

    return float(f1[max_idx].cpu().numpy()), float(recall[max_idx].cpu().numpy()), \
           float(precision[max_idx].cpu().numpy()), thresholds[max_idx]  # need np.float64 in logging rather than np.float32

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data import datasets
    from data.utils import parse_args, load_config

    args = parse_args()
    cfg = load_config(args)
    cfg['data']['use_mask'] = False

    gg = datasets.GaussianGenerator(cfg)

    # weights = prior_penalization(
    #     gg(torch.tensor([0.5, 0.5])),
    #     1-gg(torch.tensor([0.51, 0.51]))
    # )

    weights = prior_penalization(
        np.random.rand(100, 224, 224),
        np.random.rand(100, 224, 224),
    )
    print(f'{weights.shape=}')
    print(f'{weights=}')
    print(f'{weights.mean()=}')