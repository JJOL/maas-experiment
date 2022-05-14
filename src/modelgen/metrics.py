import numpy as np

def classification_scores(abs_conf_matrix, n_classes):
    metrics = {}

    mat = np.array(abs_conf_matrix)

    f1s = []
    precisions = []
    recalls = []
    for i in range(n_classes):
        TP = mat[i, i]
        FP = np.sum(mat[:, i]) - TP
        FN = np.sum(mat[i, :]) - TP

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        F1 = 2 * (precision*recall) / (precision + recall)

        f1s.append(F1)
        precisions.append(precision)
        recalls.append(recall)
    
    metrics["average_f1"] = sum(f1s) / n_classes
    metrics["average_precision"] = sum(precisions) / n_classes
    metrics["average_recall"] = sum(recalls) / n_classes


    return metrics