# src/evaluation.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

def plot_confusion_matrix(y_true, y_pred, out_png: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, cmap=None)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_roc(y_true, scores, out_png: str):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr, tpr, label='ROC')
    ax.plot([0,1], [0,1], linestyle='--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_pr(y_true, scores, out_png: str):
    prec, rec, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(rec, prec)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
