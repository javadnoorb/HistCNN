import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.integrate import trapz
import re
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import sklearn as sk
from scipy import interp
import pickle
# from itertools import cycle

get_recall = lambda confmat: confmat[1,1]/confmat[1,:].sum() if confmat[1,:].sum()>0 else np.nan # TP/(TP+FN)
get_specificity = lambda confmat: confmat[0,0]/confmat[0,:].sum() if confmat[0,:].sum()>0 else np.nan # TN/(TN+FP)
get_precision   = lambda confmat: confmat[1,1]/confmat[:,1].sum() if confmat[:,1].sum()>0 else np.nan # TP/(TP+FP)
get_accuracy    = lambda confmat: (confmat[0,0] + confmat[1,1])/confmat.sum() # (TP+TN)/(TP+TN+FP+FN)



def get_classnames(image_files_metadata, label='label'):
    class_names = image_files_metadata[[label, 'label_name']].drop_duplicates()
    class_names = class_names.sort_values(label)['label_name'].values
    return class_names


def onehotencoder(a):
    assert a.ndim == 1
    b = np.zeros((len(a), max(a)+1))
    b[np.arange(len(a)), a] = 1
    return b

def plot_multiclass_roc(ground_truth, y_score, label_names, lw=2, plot_results=True):
    y_test = onehotencoder(ground_truth)
    # Compute ROC curve and ROC area for each class


    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = sk.metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = sk.metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = sk.metrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = sk.metrics.auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sk.metrics.auc(fpr["macro"], tpr["macro"])

    if plot_results:
        # Plot all ROC curves
        plt.figure(figsize=(5, 5))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average (AUC = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average (AUC = {0:0.2f})'.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='{0} (AUC = {1:0.2f})'
                     ''.format(label_names[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='normal')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='normal')
        plt.legend(loc="lower right")
        plt.show()
    
    return roc_auc
        
def plot_pertile_prob_distributions(imagefilenames, predictions_list, final_softmax_outputs_list, image_files_metadata):
    preds = pd.DataFrame({'rel_path': imagefilenames, 'label_pred__': predictions_list[0]})
    nclass = final_softmax_outputs_list[0].shape[1]
    colname = lambda n: 'softmax_prob_{}'.format(n)
    probs = pd.DataFrame(final_softmax_outputs_list[0], columns=map(colname, range(nclass)))
    preds = preds.join(probs)
    preds['rel_path'] = preds['rel_path'].map(lambda s: s.decode())
    preds = image_files_metadata.merge(preds, on='rel_path', how='inner')

    label_names = image_files_metadata[['label_name', 'label']].drop_duplicates().set_index('label')['label_name'].values

    tmp = pd.melt(preds, id_vars=['label'], value_vars=list(probs.columns))
    tmp['variable'] = tmp['variable'].replace({'softmax_prob_{}'.format(n): label_names[n] for n in range(nclass)})
    tmp['label'] = tmp['label'].replace({n: label_names[n] for n in range(nclass)})
    tmp.rename(columns={'label': 'true label', 'variable': 'predicted label', 'value': 'classification probability'}, inplace =True)

    g = sns.violinplot(x='true label', y='classification probability', hue='predicted label', data=tmp, cut=1)
    plt.xlabel(g.get_xlabel(), fontsize=12, fontweight='normal')
    plt.ylabel(g.get_ylabel(), fontsize=12, fontweight='normal')
    g.legend(bbox_to_anchor=(1., 1), fontsize=12);

def plot_perslide_roc(predictions_df, plot_results=True):
    tmp = predictions_df.groupby(['sample_id', 'label'])['label_pred'].value_counts(normalize=True, sort=False).unstack(fill_value=0)
    tmp.reset_index(1, inplace=True)
    #label_names = predictions_df[['label_name', 'label']].drop_duplicates().set_index('label')['label_name'].values
    label_names = get_classnames(predictions_df)
    nclass = len(label_names)
    
    roc_auc = plot_multiclass_roc(tmp['label'].values, tmp[range(nclass)].values, label_names, plot_results=plot_results)
    return roc_auc

def plot_pertile_roc(imagefilenames, predictions_list, final_softmax_outputs_list, image_files_metadata, plot_results=True):
    preds = pd.DataFrame({'rel_path': imagefilenames, 'label_pred__': predictions_list[0]})
    nclass = final_softmax_outputs_list[0].shape[1]
    colname = lambda n: 'softmax_prob_{}'.format(n)
    probs = pd.DataFrame(final_softmax_outputs_list[0], columns=map(colname, range(nclass)))
    preds = preds.join(probs)
    preds['rel_path'] = preds['rel_path'].map(lambda s: s.decode())    
    preds = image_files_metadata.merge(preds, on='rel_path', how='inner')
    
    label_names = image_files_metadata[['label_name', 'label']].drop_duplicates().set_index('label')['label_name'].values
    roc_auc = plot_multiclass_roc(preds['label'].values, preds[probs.columns].values, label_names, plot_results=plot_results)
    return roc_auc

def calculate_classification_evaluation_measures(confusion_matrices_list, label_names):    
    df = pd.DataFrame(index=label_names)
    df['precision'] = list(map(get_precision, confusion_matrices_list))
    df['recall'] = list(map(get_recall, confusion_matrices_list))
    df['specificity'] = list(map(get_specificity, confusion_matrices_list))
    df['accuracy'] = list(map(get_accuracy, confusion_matrices_list))    
    return df

def plot_classification_evaluation_measures(confusion_matrices_list, label_names):
    df = calculate_classification_evaluation_measures(confusion_matrices_list, label_names)
    df.sort_values('precision').plot.barh(fontsize=16).legend(bbox_to_anchor=(1., 1), fontsize=12);
    plt.xlim([0,1]);

def plot_classification_evaluation_measures_vertical(confusion_matrices_list, label_names):    
    df = calculate_classification_evaluation_measures(confusion_matrices_list, label_names)
    ax = df.plot.bar(width=0.7, fontsize=18, figsize=(np.ceil(0.7*len(label_names)), 4))
    ax.legend(bbox_to_anchor=(1., 1), fontsize=12);
    plt.ylim([0,1]);

def plot_confusion_matrices_list(confusion_matrices_list,label_names,num_cols = 4, subplot_side_length=3):
    num_cols = min(num_cols, len(label_names))
    N = len(confusion_matrices_list)    
    num_rows = int(np.ceil(N/num_cols))
    fig,axes = plt.subplots(num_rows,num_cols,
                           figsize=(subplot_side_length*num_cols,subplot_side_length*num_rows))

    if (num_cols == 1) and (num_rows==1):
        axes = np.array([axes])
    if len(axes.shape)==1:
        axes = np.reshape(axes, (-1,axes.shape[0]))
    for n in range(N):
        plt.sca(axes[int(n/num_cols),n%num_cols])
        sns.heatmap(confusion_matrices_list[n],square=True,cmap=plt.cm.Reds,linecolor='black',lw=.01)
        plt.xlabel('Predicted Label');
        plt.ylabel('True Label');
        plt.title(label_names[n],fontweight='normal');
    
    for n in range(N,num_rows*num_cols):
        plt.sca(axes[int(n/num_cols),n%num_cols])
        plt.axis('off')
    plt.tight_layout()

def plot_all_evaluations(confusion_matrices_list,label_names, vertical=True):
    plot_confusion_matrices_list(confusion_matrices_list, label_names)
    if vertical:
        plot_classification_evaluation_measures_vertical(confusion_matrices_list, label_names)
    else:
        plot_classification_evaluation_measures(confusion_matrices_list, label_names)

def get_per_slide_average_predictions(image_files_metadata, imagefilenames, 
                                      predictions_list, label_names, final_softmax_outputs_list=None):
    assert len(label_names) == len(predictions_list)
    label_names_pred = [s+"_pred" for s in label_names]
    
    if final_softmax_outputs_list == None:
        predictions_df = pd.DataFrame([imagefilenames] + predictions_list,
                                  index=['rel_path']+label_names_pred).T
    else:
        predictions_df = pd.DataFrame([imagefilenames] + predictions_list + final_softmax_outputs_list,
                                  index=['rel_path']+label_names_pred+['pred_probs']).T
            
    predictions_df['rel_path'] = predictions_df['rel_path'].map(lambda x: x.decode())
    predictions_df[label_names_pred] = predictions_df[label_names_pred].astype(int)
    predictions_df = image_files_metadata.merge(predictions_df, how = 'right', on='rel_path')
    votes = predictions_df.groupby('sample_id')[label_names + label_names_pred].mean()
    return votes, predictions_df

def get_per_slide_evaluation_metrics(per_slide_average_predictions, per_slide_average_threshold, label):
    per_slide_predictions = per_slide_average_predictions[label+'_pred'] >= per_slide_average_threshold
    per_slide_evaluation_metrics = pd.Series(index = ['recall', 'specificity', 'precision', 'accuracy', 'false positive rate'])
    confmat = confusion_matrix(per_slide_average_predictions[label], per_slide_predictions, labels=[0,1])
    per_slide_evaluation_metrics['recall'] = get_recall(confmat)
    per_slide_evaluation_metrics['specificity'] = get_specificity(confmat)
    per_slide_evaluation_metrics['precision'] = get_precision(confmat)
    per_slide_evaluation_metrics['accuracy'] = get_accuracy(confmat)
    per_slide_evaluation_metrics['false positive rate'] = 1 - per_slide_evaluation_metrics['specificity']
    return per_slide_evaluation_metrics

def get_per_slide_evaluation_metrics_for_many_thresholds(per_slide_average_predictions, label, per_slide_average_thresholds = np.arange(0,1.002,.001)):
    per_slide_evaluation_metrics = []
    for per_slide_average_threshold in per_slide_average_thresholds:
        per_slide_evaluation_metric = get_per_slide_evaluation_metrics(per_slide_average_predictions, per_slide_average_threshold, label)
        per_slide_evaluation_metrics.append(per_slide_evaluation_metric)

    per_slide_evaluation_metrics = pd.concat(per_slide_evaluation_metrics, axis=1).T
    per_slide_evaluation_metrics['per_slide_average_threshold'] = per_slide_average_thresholds

    return per_slide_evaluation_metrics

def get_per_slide_evaluation_metrics_for_many_thresholds_for_all_labels(
    per_slide_average_predictions, label_names,
    per_slide_average_thresholds = np.arange(0,1.002,.001)
    ):
 
    per_slide_evaluation_metrics_df = []
    for label in tqdmn(label_names):
        per_slide_evaluation_metrics = get_per_slide_evaluation_metrics_for_many_thresholds(
            per_slide_average_predictions[[label, label+'_pred']], label,
            per_slide_average_thresholds = per_slide_average_thresholds)
        
        per_slide_evaluation_metrics_df.append(per_slide_evaluation_metrics.set_index('per_slide_average_threshold'))
        
    per_slide_evaluation_metrics_df = pd.concat(per_slide_evaluation_metrics_df, axis=1, keys=label_names)
    return per_slide_evaluation_metrics_df

def get_PR_AUC(per_slide_evaluation_metrics):
    temp = per_slide_evaluation_metrics.dropna(axis=0).sort_values('per_slide_average_threshold', ascending=False)
    AUC = trapz(temp['precision'], temp['recall'])
    return AUC

def get_ROC_AUC(per_slide_evaluation_metrics):
    temp = per_slide_evaluation_metrics.dropna(axis=0).sort_values('per_slide_average_threshold', ascending=False)
    AUC = trapz(temp['recall'], temp['false positive rate'])
    return AUC

def plot_PR_curve(per_slide_evaluation_metrics, pad = 0.05):
    plt.plot(per_slide_evaluation_metrics['recall'], 
            per_slide_evaluation_metrics['precision'], lw=3)
    plt.xlabel('recall', fontsize=16, fontweight='normal');
    plt.ylabel('precision', fontsize=16, fontweight='normal');   
#    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([-pad, 1+pad]);
    plt.ylim([-pad, 1+pad]);
    
    AUC = get_PR_AUC(per_slide_evaluation_metrics)   
    plt.text(1, 0.1, "PR AUC = {:.2f}".format(AUC), fontsize=18, fontweight='normal', color='red', ha='right')

def plot_ROC_curve(per_slide_evaluation_metrics, pad = 0.05):
    plt.plot(per_slide_evaluation_metrics['false positive rate'], 
            per_slide_evaluation_metrics['recall'], lw=3)
    plt.xlabel('false positive rate', fontsize=16, fontweight='normal');
    plt.ylabel('recall', fontsize=16, fontweight='normal');
    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([-pad, 1+pad]);
    plt.ylim([-pad, 1+pad]);

    AUC = get_ROC_AUC(per_slide_evaluation_metrics)   
    plt.text(1, 0.1, "ROC AUC = {:.2f}".format(AUC), fontsize=18, fontweight='normal', color='red', ha='right')
    
def plot_AUCs(image_files_metadata, imagefilenames, predictions_list, label_names, label):
    per_slide_average_predictions, _ = get_per_slide_average_predictions(
        image_files_metadata, imagefilenames, predictions_list, label_names)
    
    per_slide_evaluation_metrics = get_per_slide_evaluation_metrics_for_many_thresholds(per_slide_average_predictions[[label, label+'_pred']], label)
    per_slide_evaluation_metrics.fillna(method='ffill', inplace=True)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plot_ROC_curve(per_slide_evaluation_metrics)
    plt.subplot(1,2,2)
    plot_PR_curve(per_slide_evaluation_metrics)

def get_predicted_slide_matrix(predictions_df, sample_id, label):
    tmp = predictions_df[predictions_df['sample_id'] == sample_id].copy()
    pattern = re.compile('(\d+)_(\d+)\.jpg$')
    tmp['re_groups'] = tmp['image_filename'].map(lambda x: re.search(pattern, x))
    tmp['coord_x'] = tmp['re_groups'].map(lambda x: int(x.group(1)))
    tmp['coord_y'] = tmp['re_groups'].map(lambda x: int(x.group(2)))
    tmp = pd.pivot(tmp, index='coord_x', columns='coord_y', values=label+'_pred')
    tmp = (tmp + 1).fillna(0).astype(int)
    return tmp

def plot_predicted_slide_matrix(predictions_df, sample_id, label = 'is_tumor', figsize=(8,4),
                                labels = ['Background', 'Normal', 'Tumor'],
                                cmap = colors.ListedColormap(
                                    ["white",'green','brown'], name='from_list', N=None)
                               ):
    predicted_slide_matrix = get_predicted_slide_matrix(predictions_df, sample_id, label)
    plt.figure(figsize=figsize)
    plt.imshow(predicted_slide_matrix.T, cmap=cmap);
    cb = plt.colorbar();
    N = predicted_slide_matrix.max().max()
    dN = N/(N+1)/2
    cb.set_ticks(np.arange(dN, N, 2*dN))
    cb.set_ticklabels(labels)
    plt.xticks([]);
    plt.yticks([]);    
    
def get_PR_AUC_for_all_labels(per_slide_evaluation_metrics_df):
    tmp = per_slide_evaluation_metrics_df.dropna().sort_index(ascending=False)
    AUCs = trapz(tmp.xs('precision', axis=1, level=1).T, tmp.xs('recall', axis=1, level=1).T)
    return AUCs

def get_ROC_AUC_for_all_labels(per_slide_evaluation_metrics_df):
    tmp = per_slide_evaluation_metrics_df.dropna().sort_index(ascending=False)
    AUCs = trapz(tmp.xs('recall', axis=1, level=1).T, tmp.xs('false positive rate', axis=1, level=1).T)
    return AUCs    

def plot_PR_curve_for_all_labels(per_slide_evaluation_metrics_df, pad = 0.05):
    plt.plot(per_slide_evaluation_metrics_df.xs('recall', axis=1, level=1), 
            per_slide_evaluation_metrics_df.xs('precision', axis=1, level=1), lw=1);

    plt.xlabel('recall', fontsize=16, fontweight='normal');
    plt.ylabel('precision', fontsize=16, fontweight='normal');   
    plt.xlim([-pad, 1+pad]);
    plt.ylim([-pad, 1+pad]);
    plt.legend(per_slide_evaluation_metrics_df.columns.levels[0].tolist(), bbox_to_anchor=(1., 1), fontsize=10);
    
def plot_ROC_curve_for_all_labels(per_slide_evaluation_metrics_df, pad = 0.05):
    plt.plot(per_slide_evaluation_metrics_df.xs('false positive rate', axis=1, level=1), 
            per_slide_evaluation_metrics_df.xs('recall', axis=1, level=1), lw=1);

    plt.xlabel('false positive rate', fontsize=16, fontweight='normal');
    plt.ylabel('recall', fontsize=16, fontweight='normal');   
    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([-pad, 1+pad]);
    plt.ylim([-pad, 1+pad]);
    plt.legend(per_slide_evaluation_metrics_df.columns.levels[0], bbox_to_anchor=(1., 1), fontsize=10);
    
def barplot_AUCs(per_slide_evaluation_metrics_df):
    tmp = per_slide_evaluation_metrics_df.fillna(method='ffill')
    label_names = per_slide_evaluation_metrics_df.columns.levels[0]
    AUCs_df = pd.DataFrame([get_ROC_AUC_for_all_labels(tmp),
                            get_PR_AUC_for_all_labels(tmp)],
                           index = ['ROC_AUC', 'PR_AUC'], columns=label_names)

    AUCs_df.T.plot.bar(figsize=(13,4), fontsize=16)
    plt.ylim([0,1]);
    plt.ylabel('AUC', fontsize=18, fontweight='normal');
    plt.plot([-0.5, len(label_names)], [0.5, 0.5], 'r--');    
    

def get_per_slide_probs(cancertype, pickle_path):
    picklefile = '{:s}/run_cnn_output_{:s}.pkl'.format(pickle_path, cancertype)
    label_names = ['is_tumor']
    [image_files_metadata, test_accuracies_list, predictions_list, confusion_matrices_list, imagefilenames, final_softmax_outputs_list] = pickle.load(open(picklefile, 'rb'))
    label = label_names[0]
    per_slide_average_predictions, _ = get_per_slide_average_predictions(
        image_files_metadata, imagefilenames, predictions_list, label_names)
    return per_slide_average_predictions

def get_per_slide_probs_pancan(pickle_path, cancertypes):
#     cancertypes = [s.split('/')[-1][15:-4] for s in glob.glob(pickle_path+'/run_cnn_output_*.pkl')] 
    per_slide_probs = pd.DataFrame()

    pbar = tqdm(cancertypes)

    for cancertype in pbar:
        pbar.set_description("Processing %s" % cancertype)
        tmp = get_per_slide_probs(cancertype, pickle_path)
        tmp['cancertype'] = cancertype.upper()
        per_slide_probs = pd.concat([per_slide_probs, tmp])

    per_slide_probs.to_csv(pickle_path + '/per_slide_probs.txt')
    return per_slide_probs

def plot_per_slide_probs_pancan(pickle_path, sorted_list, violinplot=False, ax=None):
    per_slide_probs = pd.read_csv(pickle_path + '/per_slide_probs.txt', index_col=0)
#     AUCs = pd.read_csv(pickle_path+'/AUCs.txt', index_col=0)
#     AUCs.sort_values('ROC AUC', ascending=False, inplace=True)
    per_slide_probs['is_tumor'].replace({0: 'Normal', 1:'Tumor'}, inplace=True)
    if violinplot:
        sns.violinplot(x='cancertype', y='is_tumor_pred', hue='is_tumor', data=per_slide_probs,
                       hue_order = ['Normal', 'Tumor'], inner=None, scale='width', order = sorted_list.str.upper(),
                       ax=ax, palette=['lightgreen', 'darkgreen'], split=True)
    else:
        sns.barplot(x='cancertype', y='is_tumor_pred', hue='is_tumor', data=per_slide_probs, palette=['lightgreen', 'darkgreen'],
                         hue_order = ['Normal', 'Tumor'], order = sorted_list.str.upper(), ax=ax)
    
    ax.legend().set_title('')
    ax.set_ylabel('predicted tumor fraction\nper slide', fontsize=12, fontweight='normal');
    ax.set_xlabel('');
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='center')

def get_AUC(cancertype, pickle_path):
    picklefile = '{:s}/run_cnn_output_{:s}.pkl'.format(pickle_path, cancertype)
    label_names = ['is_tumor']
    [image_files_metadata, test_accuracies_list, predictions_list,
     confusion_matrices_list, imagefilenames, final_softmax_outputs_list] = pickle.load(open(picklefile, 'rb'))

    label = 'is_tumor'
    per_slide_average_predictions, _ = get_per_slide_average_predictions(
        image_files_metadata, imagefilenames, predictions_list, label_names)

    per_slide_evaluation_metrics = get_per_slide_evaluation_metrics_for_many_thresholds(per_slide_average_predictions[[label, label+'_pred']], label)
    per_slide_evaluation_metrics.fillna(method='ffill', inplace=True)
    AUC = get_ROC_AUC(per_slide_evaluation_metrics)
    prAUC = get_PR_AUC(per_slide_evaluation_metrics)
    return AUC, prAUC

def get_AUCs_pancan(pickle_path, cancertypes):
#     cancertypes = [s.split('/')[-1][15:-4] for s in glob.glob(pickle_path+'/run_cnn_output_*.pkl')] 
    AUCs = pd.DataFrame(index=cancertypes, columns=['ROC AUC', 'PR AUC'])

    pbar = tqdm(cancertypes)
    for cancertype in pbar:
        pbar.set_description("Processing %s" % cancertype)
        x = get_AUC(cancertype, pickle_path)
        AUCs.loc[cancertype, 'ROC AUC'] = x[0]
        AUCs.loc[cancertype, 'PR AUC'] = x[1]
    AUCs.to_csv(pickle_path+'/AUCs.txt')    
    return AUCs

def plot_AUCs_pancan(pickle_path, sorted_list, ax=None, ylim=[0.97, 1]):
    AUCs = pd.read_csv(pickle_path+'/AUCs.txt', index_col=0)
    AUCs = AUCs[(AUCs>0).all(axis=1)]
    AUCs.index = AUCs.index.map(str.upper)

#     AUCs.sort_values('ROC AUC', ascending=False, inplace=True)
    AUCs = AUCs.loc[sorted_list.str.upper()]

    AUCs.plot.bar(color=['lightgreen', 'darkgreen'], ax=ax, log=False);
    ax.legend([x.split()[0] for x in AUCs.columns], loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.set_ylabel('AUC', fontsize=12, fontweight='normal');
    ax.set_xlabel('')
    ax.set_ylim(ylim)
    
def get_metrics(cancertype, pickle_path):
    picklefile = '{:s}/run_cnn_output_{:s}.pkl'.format(pickle_path, cancertype)
    label_names = ['is_tumor']
    [image_files_metadata, test_accuracies_list, predictions_list, confusion_matrices_list, imagefilenames, final_softmax_outputs_list] = pickle.load(open(picklefile, 'rb'))
    metrics = calculate_classification_evaluation_measures(confusion_matrices_list, label_names)
    return metrics

def get_metrics_pancan(pickle_path, cancertypes):
#     cancertypes = [s.split('/')[-1][15:-4] for s in glob.glob(pickle_path+'/run_cnn_output_*.pkl')] 
    metrics = pd.DataFrame()

    pbar = tqdm(cancertypes)

    for cancertype in pbar:
        pbar.set_description("Processing %s" % cancertype)
        tmp = get_metrics(cancertype, pickle_path)
        tmp = tmp.T.reset_index().rename(columns={'index':'metric'})
        tmp['cancertype'] = cancertype.upper()
        metrics = pd.concat([metrics, tmp], sort=False)

    metrics.to_csv(pickle_path+'/metrics.txt')
    return metrics

def plot_metrics_pancan(pickle_path, sorted_list, ax=None, ylim=[0.7, 1]):
    metrics = pd.read_csv(pickle_path+'/metrics.txt', index_col=0)
    metrics['metric'] = metrics['metric'].str.capitalize()
#     AUCs = pd.read_csv(pickle_path+'/AUCs.txt', index_col=0)
#     AUCs.sort_values('ROC AUC', ascending=False, inplace=True)
#     AUCs = AUCs.loc[sorted_list]
    sns.barplot(x='cancertype', y='is_tumor', hue='metric', data=metrics, order = sorted_list.str.upper(),
                     palette=['lightgreen', 'limegreen', 'green', 'darkgreen'], ax=ax)
    ax.legend().set_title('')
    ax.set_ylabel('per-tile\nclassification metric', fontsize=12, fontweight='normal');
    ax.set_xlabel('');
    ax.set_ylim(ylim)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='center')