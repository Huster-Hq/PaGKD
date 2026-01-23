from sklearn.metrics import roc_curve
import argparse
import os
import seaborn as sns
import pandas as pd
from itertools import cycle
from utils.loader import PolyDataset_nocrop_5fold_pair

from models.model_stu import Resnet_att
import numpy as np
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            accuracy_score, roc_auc_score, roc_curve, 
                            confusion_matrix, classification_report,
                            precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize


def load_model(i,model,model_num):
    pth_path=os.path.join(opt.pth_path,str(i),'weights',model_num)
    print(model.load_state_dict(torch.load(pth_path, map_location=opt.device),strict=False))
    model = model.student
    return model
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

parser.add_argument('--pth_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='', help='save_path')
opt = parser.parse_args()
if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)


all_predict_list=[]
all_predict_list0=[]
all_label_list=[]

num_list=[]
right_label_list=[]
wrong_label_list=[]
wrong_image_list=[]
data_path=[]
class_names = ['Hyperplasia','Adenoma','Adenocarcinoma'] 


plt.figure(figsize=(10, 8))

y_true, y_pred, y_score = [], [], []

for i in range(5):
    nu=[16,63,98,18,31]
    model_num='poly_student_model-'+str(nu[i])+'.pth'

    model = Resnet_att(num_classes=opt.num_classes).to(opt.device)
    model.eval()
    model=load_model(i,model,model_num)
    val_dataset = PolyDataset_nocrop_5fold_pair(is_train=False,split_id=i)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             )
    
    with torch.no_grad():
        for batch in tqdm(val_loader):

            inputs,  labels, wht_path = batch[0].to(opt.device).float(),batch[2].to(opt.device).long(),batch[3]
            
            outputs,_,_ = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)

y_true_bin = label_binarize(y_true, classes=np.arange(opt.num_classes))

accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro')
recall_macro = recall_score(y_true, y_pred, average='macro')
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_macro = f1_score(y_true, y_pred, average='macro')


precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
    y_true, y_pred, average=None, labels=np.arange(opt.num_classes))


conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(opt.num_classes))
specificity_per_class = []
for i in range(opt.num_classes):
    tn = np.sum(np.delete(np.delete(conf_mat, i, axis=0), i, axis=1))
    fp = np.sum(np.delete(conf_mat[i, :], i))
    specificity_per_class.append(tn / (tn + fp))


auc_scores = []
for i in range(opt.num_classes):
    auc_scores.append(roc_auc_score(y_true_bin[:, i], y_score[:, i]))
macro_auc = np.mean(auc_scores)


overall_metrics = {
    'Metric': ['Accuracy', 'Precision (macro)', 'Recall (macro)', 
                'Specificity (macro)', 'F1-score (macro)','F1-score (macro)', 'AUC (macro)'],
    'Value': [accuracy, precision_macro, recall_macro, 
                np.mean(specificity_per_class), f1_macro,f1_macro, macro_auc]
}
overall_df = pd.DataFrame(overall_metrics)
overall_df.to_csv(os.path.join(opt.save_path, 'overall_metrics.csv'), index=False)

class_metrics = {
    'Class': class_names,
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'Specificity': specificity_per_class,
    'F1-score': f1_per_class,
    'AUC': auc_scores,
    'Support': support
}
class_df = pd.DataFrame(class_metrics)
class_df.to_csv(os.path.join(opt.save_path, 'class_metrics.csv'), index=False)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(opt.save_path, 'confusion_matrix.png'))
plt.close()
plt.figure(figsize=(8, 6))


colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(opt.num_classes), colors):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, color=color, lw=2,
                label='ROC curve of {0} (AUC = {1:0.2f})'
                ''.format(class_names[i], auc_scores[i]))


all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_score[:, i])[0] 
                                    for i in range(opt.num_classes)]))


mean_tpr = np.zeros_like(all_fpr)
for i in range(opt.num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    mean_tpr += np.interp(all_fpr, fpr, tpr)

mean_tpr /= opt.num_classes

macro_auc = roc_auc_score(y_true_bin, y_score, multi_class='ovr', average='macro')

plt.plot(all_fpr, mean_tpr, color='deeppink', linestyle=':', linewidth=4,
            label='Macro-average ROC curve (AUC = {0:0.3f})'
            ''.format(macro_auc))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(opt.save_path, 'roc_curve.png'))
plt.close()

print("Evaluation completed. Results saved to", opt.save_path)