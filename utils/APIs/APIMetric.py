from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, classification_report
)

def api_metric(true_labels, pred_labels, pos_label=1, verbose=True):
    # 安全：固定标签顺序，避免某一类缺失导致顺序乱
    labels = [0, 1]
    target_names = ['negative', 'positive']

    if verbose:
        print(classification_report(
            true_labels, pred_labels,
            labels=labels, target_names=target_names,
            digits=4, zero_division=0
        ))

    acc = accuracy_score(true_labels, pred_labels)
    # 二分类 F1（以 pos_label 为正类）
    f1_bin = f1_score(true_labels, pred_labels, average='binary', pos_label=pos_label, zero_division=0)
    # 宏平均（不看类别比例）
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, average='macro', zero_division=0
    )
    # 加权平均（按支持度加权）
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, average='weighted', zero_division=0
    )

    return {
        "accuracy": acc,
        "f1_binary": f1_bin,       # 以正类为基准的 F1
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,      # 你要的“宏指标”
        "f1_weighted": f1_weighted
    }
