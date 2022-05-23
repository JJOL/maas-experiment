import numpy as np

def classification_scores(conf_matrix, n_classes):
    """
    Returns a dictionary of the results of average F1, average precision and average recall of the categories in the given confussion matrix.
    """
    metrics = {}

    conf_matrix = np.array(conf_matrix)

    f1s = []
    precisions = []
    recalls = []
    for i in range(n_classes):
        TP = conf_matrix[i, i]
        FP = np.sum(conf_matrix[:, i]) - TP
        FN = np.sum(conf_matrix[i, :]) - TP

        # Check for cases when divisions by 0 occur to ignore them as results=0
        if TP == 0:
            precision = 0
            recall = 0
            F1 = 0
        else:
            # Metric Formulas: https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/#:~:text=F1%20score%20%2D%20F1%20Score%20is,have%20an%20uneven%20class%20distribution.
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