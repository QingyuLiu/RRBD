import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_recall_curve, roc_curve, auc


def calculate_metrics(true_labels,pred,one_class=False):
    pred_labels = pred>0.5
    base_rate = 0.6
    cm = confusion_matrix(true_labels, pred_labels)
    TN, FP, FN, TP = cm.ravel()
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    if one_class:
        return TPR,FPR, 0,  0,0,0,0,0
    FDR = FP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = recall_score(true_labels, pred_labels)
    ACC = accuracy_score(true_labels, pred_labels)

    # print(f"FP:{FP}")
    # print(f"FN:{FN}")
    # print(f"TP:{TP}")
    # print(f"TN:{TN}")

    precision, Recall1, _ = precision_recall_curve(true_labels, pred)
    fpr, tpr, thresholds = roc_curve(true_labels, pred)
    auprc = auc(Recall1, precision)
    auroc = auc(fpr, tpr)
    bdr = TPR*base_rate/(base_rate*TPR+(1-base_rate)*FPR)
    # print(bdr)
    fnr = 1 - tpr

    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    return TPR,FPR, FDR,  ACC,auprc,auroc,bdr,eer


def calculate_tpr_at_fpr(predictions, labels, target_fpr):

    sorted_indices = np.argsort(-predictions)
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]


    n_negative = np.sum(labels == 0)
    fp_allowed = int(np.ceil(target_fpr * n_negative))


    fp = 0
    tp = 0
    for i, label in enumerate(sorted_labels):
        if label == 0:
            fp += 1
        else:
            tp += 1


        if fp == fp_allowed:
            tpr = tp / np.sum(labels == 1)
            return tpr

    return 0
