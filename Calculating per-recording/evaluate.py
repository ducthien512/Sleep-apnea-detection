import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "Sleep Apnea detection"

# Table
output = []
methods = ["CNN-Transformer-LSTM"]
for method in methods:
    df = pd.read_csv(os.path.join(base_dir, "%s.csv" % method), header=0)
    df["y_pred"] = df["y_score"] > 0.5
    df = df.groupby(by="subject").apply(lambda d: d["y_pred"].mean() * 60)
    df.name = method
    output.append(df)
output = pd.concat(output, axis=1)
with open("dataset/additional-information.txt", "r") as f:
    original = []
    for line in f:
        rows = line.strip().split("\t")
        if len(rows) == 12:
            if rows[0].startswith("x"):
                original.append([rows[0], float(rows[3]) / float(rows[1]) * 60])
original = pd.DataFrame(original, columns=["subject", "original"])
original = original.set_index("subject")
all = pd.concat((output, original), axis=1)
corr = all.corr()
all1 = all.applymap(lambda a: int(a > 5))
result = []
for method in methods:
    C = confusion_matrix(all1["original"], all1[method], labels=(0, 1))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    auc = roc_auc_score(all["original"] > 5, all[method])
    result.append([method, acc * 100, sn * 100, sp * 100, auc, corr["original"][method]])
np.savetxt(os.path.join(base_dir, "Table.csv"), result, fmt="%s", delimiter=",", comments="",
           header="Method,Accuracy(%),Sensitivity(%),Specificity(%),AUC,Corr")

# Define labels for confusion matrix
labels = ['Non-Apnea', 'Apnea']

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(C, annot=True, cmap='Reds', fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('Confusion Matrix_1.png', bbox_inches='tight', dpi=300)
plt.show()
