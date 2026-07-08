import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    N = max(max(y_true), max(y_pred)) + 1
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    
    confusion_matrix = np.array([[(np.sum((y_true==j) & (y_pred==i))) for i in range(N)] for j in range(N)])


    betta = 1
    F_betta = lambda b, p, r: (1+b**2) * (p*r)/(b**2 * p + r)


    
    glob = np.zeros(4) #global TP, FN, FP, TN
    
    precision, recall, f1 = [], [], []
    
    for i in range(N):
        TP = confusion_matrix[i, i]
        FN = np.sum(confusion_matrix[i, :]) - confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
        TN = np.sum(confusion_matrix) - FN - FP

        glob += np.array([TP, FN, FP, TN])


        acc = (TP + TN)/(TP + TN + FP + FN)
        pr = TP / (TP+FP)
        rec = TP / (TP+FN)
        F_1 = F_betta(betta, pr, rec)

        precision.append(pr)
        recall.append(rec)
        f1.append(F_1)

    global_accuracy = np.mean(y_true == y_pred)
        
    if average == "micro":
        precision = glob[0]/(glob[0]+glob[2])
        recall = glob[0]/(glob[0]+glob[1])
        f_1 = F_betta(1, precision, recall)
        return {"accuracy": float(global_accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f_1)}

    if average == "macro":
        precision, recall, f1 = np.array(precision), np.array(recall), np.array(f1)
        
        return {"accuracy": float(global_accuracy), "precision": float(np.mean(precision)), "recall": np.mean(recall), "f1": float(np.mean(f1))}

    if average == "weighted":
        weights = np.array([np.sum(y_true == i) for i in range(N)])
        sum_weights = np.sum(weights)

        precision, recall, f1 = np.array(precision), np.array(recall), np.array(f1)
        weighted_precision = np.sum(weights * precision) / sum_weights
        weighted_recall = np.sum(weights * recall) / sum_weights
        weighted_f1 = np.sum(weights * f1) / sum_weights
        return {"accuracy": float(global_accuracy), "precision": float(weighted_precision), "recall": float(weighted_recall), "f1": float(weighted_f1)}
    
    if average == "binary":
        TP = confusion_matrix[pos_label, pos_label]
        FN = np.sum(confusion_matrix[pos_label, :]) - confusion_matrix[pos_label, pos_label]
        FP = np.sum(confusion_matrix[:, pos_label]) - confusion_matrix[pos_label, pos_label]
        TN = np.sum(confusion_matrix) - FN - FP

        pr = TP / (TP+FP)
        rec = TP / (TP+FN)
        F_1 = F_betta(betta, pr, rec)
        return {"accuracy": float(global_accuracy), "precision": float(pr), "recall": float(rec), "f1": float(F_1)}
        



            
                

    



    