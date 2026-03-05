import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, precision_score, recall_score, accuracy_score
import sys

def find_best_thresholds(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    labels = df['label'].values
    dcs = df['dc'].values

    # Determine range of thresholds
    min_dc = np.min(dcs)
    max_dc = np.max(dcs)
    thresholds = np.linspace(min_dc - 0.1, max_dc + 0.1, 2000)

    best_f1 = -1
    best_f1_threshold = -1
    best_mcc = -1
    best_mcc_threshold = -1

    for t in thresholds:
        preds = (dcs > t).astype(int)
        
        # We need both classes to be present in true labels and predictions for valid MCC and F1
        if len(np.unique(preds)) == 1:
            continue
            
        f1 = f1_score(labels, preds, zero_division=0)
        mcc = matthews_corrcoef(labels, preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = t
            
        if mcc > best_mcc:
            best_mcc = mcc
            best_mcc_threshold = t

    print(f"--- F1 Optimization ---")
    print(f"Best Threshold for F1: {best_f1_threshold:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    print(f"\n--- MCC Optimization ---")
    print(f"Best Threshold for MCC: {best_mcc_threshold:.4f}")
    print(f"Best MCC Score: {best_mcc:.4f}")

    # Evaluate at some chosen thresholds to show the trade-off
    print("\n--- Detailed Evaluation at Key Thresholds ---")
    
    eval_thresholds = [0.2012, best_f1_threshold, 1.0, 1.3203]
    eval_names = ["Old Optimal (0.2012)", "New F1 Optimal", "Heuristic (1.0)", "Old High Precision (1.3203)"]

    print(f"{'Threshold Name':<30} | {'Thr':<6} | {'F1':<6} | {'MCC':<6} | {'Prec':<6} | {'Rec':<6} | {'Acc':<6} | {'FP':<4} | {'FN':<4}")
    print("-" * 100)
    
    for name, t in zip(eval_names, eval_thresholds):
        preds = (dcs > t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        mcc = matthews_corrcoef(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        accuracy = accuracy_score(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        
        print(f"{name:<30} | {t:<6.4f} | {f1:<6.4f} | {mcc:<6.4f} | {precision:<6.4f} | {recall:<6.4f} | {accuracy:<6.4f} | {fp:<4} | {fn:<4}")

if __name__ == "__main__":
    csv_path = "dc_stats.csv" if len(sys.argv) == 1 else sys.argv[1]
    find_best_thresholds(csv_path)
