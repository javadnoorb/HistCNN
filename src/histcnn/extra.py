import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openslide as ops  # installed by using docker (docker exec -it [container-id] bash) and then following this : http://openslide.org/download/
import numpy as np
import tensorflow.compat.v1 as tf
import seaborn as sns
import pickle
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from histcnn import (util,
                 process_files,
                 tile_image,
                 preprocess_image,
                 run_classification,
                 plotting_cnn,
                 choose_input_list,
                 handle_tfrecords,
                 visualize_slide,
                 annotate)

from histcnn import inception_multitasklearning_retrain as incret
import random
import json

import matplotlib.colors as clr
from scipy.cluster.hierarchy import linkage, fcluster
import pickle
import pandas as pd

def plot_crossclassification_heatmap(picklefile):
    aucs = pickle.load(open(picklefile, 'rb'))
    aucs = pd.Series(aucs)

    aucs = aucs.to_frame().reset_index()
    aucs.rename(columns={0:'AUC'}, inplace=True)

    # cancertypes = ['lihc', 'paad', 'prad', 'esca',
    #                'stad', 'coad', 'read', 'oc',
    #                'ucec', 'blca', 'brca', 'sarc',
    #                'thca', 'kirp', 'hnsc', 'lusc',
    #                'luad', 'kich', 'kirc']

    aucs = pd.concat([aucs, aucs['index'].str.upper().str.split('-').apply(pd.Series)], axis=1).rename(columns={0:'TRAIN', 1:'TEST'})#.drop(['index'])

    aucs = pd.pivot(index='TRAIN', columns='TEST', values='AUC', data=aucs)
    cmap = 'RdYlBu_r'

    samplecolors = pd.DataFrame('whitesmoke', index=aucs.index, columns=['organ'])

    colors_dict = {'Lung': 'darkblue',
                   'Pan-Kidney': 'royalblue',
                   'Pan-GI': 'deepskyblue',
                   'Pan-GYN': 'lightblue',
                   'Other': 'whitesmoke'}

    # legend_colors = pd.Series(colors_dict)

    samplecolors.loc['LUAD', 'organ'] = colors_dict['Lung']
    samplecolors.loc['LUSC', 'organ'] = colors_dict['Lung']

    samplecolors.loc['KIRC', 'organ'] = colors_dict['Pan-Kidney']
    samplecolors.loc['KICH', 'organ'] = colors_dict['Pan-Kidney']
    samplecolors.loc['KIRP', 'organ'] = colors_dict['Pan-Kidney']

    samplecolors.loc['READ', 'organ'] = colors_dict['Pan-GI'] # https://doi.org/10.1016/j.cell.2018.03.022
    samplecolors.loc['COAD', 'organ'] = colors_dict['Pan-GI']
    samplecolors.loc['STAD', 'organ'] = colors_dict['Pan-GI']
    samplecolors.loc['ESCA', 'organ'] = colors_dict['Pan-GI']

    samplecolors.loc['OV', 'organ'] = colors_dict['Pan-GYN']
    samplecolors.loc['BRCA', 'organ'] = colors_dict['Pan-GYN']
    samplecolors.loc['UCEC', 'organ'] = colors_dict['Pan-GYN']

    samplecolors.loc['LIHC', 'organ'] = colors_dict['Other']
    samplecolors.loc['PAAD', 'organ'] = colors_dict['Other']
    samplecolors.loc['PRAD', 'organ'] = colors_dict['Other']
    samplecolors.loc['BLCA', 'organ'] = colors_dict['Other']
    samplecolors.loc['SARC', 'organ'] = colors_dict['Other']
    samplecolors.loc['THCA', 'organ'] = colors_dict['Other']
    samplecolors.loc['HNSC', 'organ'] = colors_dict['Other']

    colors_dict.pop('Other')
    colors_dict.update({'Adenocarcinoma': 'darkgreen',
                        'Carcinoma (non-adeno)': 'lightgreen',
                        'Other': 'whitesmoke'})

    legend_colors = pd.Series(colors_dict)
    # legend_colors = pd.concat([legend_colors, pd.Series(colors_dict)])
    # legend_colors.drop_duplicates(inplace=True, keep='last')
    # adenocarcinomas = ['LUAD', 'PAAD', 'COAD', 'PRAD', 'STAD', 'OV', 'READ', 'LIHC']
    # carcinomas = ['BLCA', 'THCA', 'KIRP', 'KICH', 'UCEC', 'LUSC', 'HNSC', 'BRCA', 'KIRC']

    adenocarcinomas = ['LUAD', 'PAAD', 'COAD', 'PRAD', 'STAD', 'OV', 'READ', 'LIHC', 'BRCA', 'KIRC']
    carcinomas = ['BLCA', 'THCA', 'KIRP', 'KICH', 'UCEC', 'LUSC', 'HNSC']
    other = ['ESCA', 'SARC']
    samplecolors.loc[adenocarcinomas, 'adeno.'] = colors_dict['Adenocarcinoma']
    samplecolors.loc[carcinomas, 'adeno.'] = colors_dict['Carcinoma (non-adeno)']
    samplecolors.loc[other, 'adeno.'] = colors_dict['Other']

    def plot_clustermap(aucs, samplecolors, legend_colors, method='average', metric='euclidean',
                        row = 'row', col = 'col', optimal_ordering=False):
        Zrow = linkage(aucs, method=method, metric=metric, optimal_ordering=optimal_ordering)
        Zcol = linkage(aucs.T, method=method, metric=metric, optimal_ordering=optimal_ordering)
        Z = {'row': Zrow, 'col': Zcol}

        g = sns.clustermap(aucs, row_linkage=Z[row], col_linkage=Z[col], cmap=cmap, vmax=1, vmin=0,
                           figsize=(len(aucs.columns)/2, len(aucs.index)/2), row_colors=samplecolors, col_colors=samplecolors)

        rect = lambda color: plt.Rectangle((0, 0), 1, 1, fc=color, fill=True, edgecolor='black', linewidth=1)
        plt.legend(legend_colors.map(rect).tolist(),
                  legend_colors.index.tolist(), loc = 'upper left',
                  bbox_to_anchor=(20, 1.1))
        return (Zrow, Zcol)

    from scipy.cluster.hierarchy import dendrogram
    import libpysal
    from esda.gamma import Gamma

    def get_gamma_index_p_value(feature, linkage_matrix, samplecolors, cancertypes, colors_dict, drop_Other=True):
        '''
        This function calculates gamma index of spatial autocorrelation for tissue feature labels
        For further information refer to:
        Hubert, Lawrence James, Reg G. Golledge, and Carmen M. Costanzo.
        "Generalized procedures for evaluating spatial autocorrelation."
        Geographical Analysis 13.3 (1981): 224-233.
        '''
        cluster_ordering = dendrogram(linkage_matrix, no_plot=True)['ivl']
        cluster_ordering = [cancertypes[int(idx)].upper() for idx in cluster_ordering]

        colors_dict_revert = {v:k for k,v in colors_dict.items()}
        tmp = samplecolors[feature].map(lambda x: colors_dict_revert[x])

        tmp = tmp.loc[cluster_ordering]
        if drop_Other:
            tmp = tmp[tmp != 'Other']
        tmp = tmp.astype('category').cat.codes

        w = libpysal.weights.lat2W(1,len(tmp))
        g = Gamma(tmp.values, w, operation = lambda y,i,j: y[i]==y[j])
        p_value = g.p_sim_g
        return p_value

    def get_all_gamma_index_p_values(Zrow, Zcol, samplecolors, cancertypes, colors_dict, drop_Other=True):
        p_values = {}
        p_values['adeno_col'] = get_gamma_index_p_value('adeno.', Zcol, samplecolors, cancertypes, colors_dict, drop_Other=drop_Other)
        p_values['adeno_row'] = get_gamma_index_p_value('adeno.', Zrow, samplecolors, cancertypes, colors_dict, drop_Other=drop_Other)
        p_values['organ_col'] = get_gamma_index_p_value('organ', Zcol, samplecolors, cancertypes, colors_dict, drop_Other=drop_Other)
        p_values['organ_row'] = get_gamma_index_p_value('organ', Zrow, samplecolors, cancertypes, colors_dict, drop_Other=drop_Other)
        return p_values

    (Zrow, Zcol) = plot_clustermap(aucs, samplecolors, legend_colors, method='average', row = 'row', col = 'col', optimal_ordering=True)
    plt.savefig(picklefile+'.crossclassificationheatmap.pdf', bbox_inches='tight')

    cancertypes = aucs.index.tolist()
    p_values = get_all_gamma_index_p_values(Zrow, Zcol, samplecolors, cancertypes, colors_dict, drop_Other=True)
    return p_values

def plot_subtype_counts(counts_file, ax=None, plot=True):
    counts = pd.read_csv(counts_file)
    counts = counts[counts['count-method'] == 'slide']

    counts['crossval_group'] = counts['crossval_group'].replace({'validation':'testing'})
    counts = counts.groupby(['crossval_group', 'tissue-method', 'tissue'])['counts'].sum().reset_index()

    tmp = counts.groupby(['tissue', 'tissue-method'])['counts'].sum().unstack(fill_value=0)

    sorted_list = counts.groupby('tissue')['counts'].sum().sort_values(ascending=False).index

    if plot:
        tmp1 = counts.pivot_table(index='tissue', columns=['crossval_group' , 'tissue-method'], values='counts')
        tmp1 = tmp1.loc[sorted_list]
        tmp1.index = tmp1.index.str.capitalize()


        tmp1.plot.bar(stacked=True, ax=ax,#figsize=(8, 4),
                      log=False, color=['lightgreen', 'darkgreen', 'red', 'darkred'])

        col0 = tmp1.columns.get_level_values(0)
        col1 = tmp1.columns.get_level_values(1)

        legend = ['{:s} ({:s})'.format(y, x[:-3]) for x,y in zip(col0, col1)]
    #     ax = plt.gca()
        ax.legend(legend, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        ax.set_ylabel('Number of\nSlides', fontsize=16, fontweight='normal');
        ax.set_ylim([0.9, 2750]);
        ax.set_xlabel('')
        ax.xaxis.set_tick_params(labelsize=14)
        # ax.yaxis.set_tick_params(labelsize=12)
    return sorted_list.tolist()


import copy

def add_bar(ax, x, y, label, color = 'lightblue', ec = 'black',
            secondarycolor='darkgreen', w = 0.05, label_offset=0.1, set_xlabels=True):
    if label in ['micro-average','macro-average']:
        color = secondarycolor
    ax.add_patch(plt.Rectangle((x, 0), w, y, alpha=1, edgecolor=ec, facecolor=color))
    if set_xlabels:
        ax.text(x + w/2, -label_offset, label, rotation=90, va='top',
                ha='center', fontsize=14, fontweight='normal')

def group_barplot(ax, y_dict, tissue, x, color = 'lightblue', secondarycolor='darkgreen',
                  ec = 'black', w = 0.05, tissue_offset=0.01, label_offset=0.1, set_xlabels=True):
    for label, auc in y_dict.items():
        x += w
        add_bar(ax, x, auc, label, color=color, ec=ec, w=w,
                secondarycolor=secondarycolor, label_offset=label_offset, set_xlabels=set_xlabels)
#         ax.plot([x, x+w], [y_dict['micro-average'], y_dict['micro-average']], 'r.', linewidth=2)
    if set_xlabels:
        ax.text(x - len(y_dict)*w/2+w, -tissue_offset, tissue.upper(),
                rotation=0, va='top', ha='center', color='red',
                fontsize=12, fontweight='normal')

def plot_all_aucs(aucs, ylabel='AUC', w = 1, bargaps_factor=2, show_legend = True,
                  tissue_offset=0.01, label_offset=0.1, ax=None, set_xlabels=True):
#     fig, ax = plt.subplots(figsize=(17,3))

    bargaps = w*bargaps_factor
    x = 0
    x_midpoints = []
    aucs_copy = copy.deepcopy(aucs)
    for tissue, aucvals in aucs_copy.items():
        if aucvals.get('micro-average') or aucvals.get('macro-average'):
            y_micro = aucvals.pop('micro-average')
            y_macro = aucvals.pop('macro-average')

            x_midpoint = x + len(aucvals)*w/2+w
            ax.plot([x_midpoint], [y_micro], 'rx')
            ax.plot([x_midpoint], [y_macro], 'ro')

            x_midpoints.append(x_midpoint)

            if show_legend:
                ax.legend(['micro-average', 'macro-average'], loc='upper left', bbox_to_anchor=(1.0, 1.0))

        group_barplot(ax, aucvals, tissue, x, w=w, color = 'lightgreen', ec = 'black',
                      secondarycolor='darkgreen', tissue_offset=tissue_offset,
                      label_offset=label_offset, set_xlabels=set_xlabels)
        x += len(aucvals) * w + bargaps

    ax.set_xlim([0, x]);
    ax.set_ylim(bottom=0.);
    ax.set_xticks([]);
    ax.set_ylabel(ylabel, fontsize=16, fontweight='normal');

    return x_midpoints

# _, ax = plt.subplots(figsize=(17,3))
# plot_all_aucs(aucs['pertile'], ax=ax)


def filter_run_results(image_files_metadata, predictions_list,
                       imagefilenames, final_softmax_outputs_list,
                       tissue_method = ['FFPE']):
    tiles_subset = image_files_metadata[image_files_metadata['tissue-method'].isin(tissue_method)]
    tiles_subset = tiles_subset['rel_path'].map(lambda s: s.encode())
    tiles_subset.reset_index(drop=True, inplace=True)

    run_df = pd.DataFrame()
    run_df['predictions_list'] = predictions_list[0]
    run_df['imagefilenames'] = imagefilenames

    final_softmax_outputs_list_colnames = []
    for n in range(final_softmax_outputs_list[0].shape[1]):
        final_softmax_outputs_list_colname = 'final_softmax_outputs_list_{:d}'.format(n)
        final_softmax_outputs_list_colnames.append(final_softmax_outputs_list_colname)
        run_df[final_softmax_outputs_list_colname] = final_softmax_outputs_list[0][:,n]

    run_df = pd.merge(run_df, tiles_subset.to_frame(), left_on='imagefilenames', right_on='rel_path')

    final_softmax_outputs_list = [run_df[final_softmax_outputs_list_colnames].values]
    imagefilenames = run_df['imagefilenames'].values
    predictions_list = [run_df['predictions_list'].values]

    return predictions_list, imagefilenames, final_softmax_outputs_list


def get_subtype_aucs_from_pickles(pickle_path, tissues, pickel_prefix = 'run_cnn_output_', tissue_method=None):
    aucs = {'perslide':{}, 'pertile':{}}

    pbar = tqdm(tissues)
    for tissue in pbar:
        picklefile = os.path.join(pickle_path, '{}{}.pkl'.format(pickel_prefix, tissue))

        pbar.set_description("%s: load data" % tissue)

        [image_files_metadata, test_accuracies_list,
         predictions_list, confusion_matrices_list,
         imagefilenames, final_softmax_outputs_list] = pickle.load(open(picklefile, 'rb'))

        if tissue_method!=None:
            predictions_list, imagefilenames, final_softmax_outputs_list = filter_run_results(image_files_metadata, predictions_list,
                                                                                      imagefilenames, final_softmax_outputs_list,
                                                                                      tissue_method = tissue_method)

        pbar.set_description("%s: calculate per tile AUCs" % tissue)
        class_names = plotting_cnn.get_classnames(image_files_metadata)
        auc = plotting_cnn.plot_pertile_roc(imagefilenames, predictions_list, final_softmax_outputs_list, image_files_metadata, plot_results=False)

        # use subtype names
        for n in range(len(class_names)):
            auc[class_names[n]] = auc.pop(n)
        aucs['pertile'][tissue] = auc

        pbar.set_description("%s: calculate per slide AUCs" % tissue)
        votes, predictions_df = plotting_cnn.get_per_slide_average_predictions(image_files_metadata, imagefilenames, predictions_list, ['label'])
        auc = plotting_cnn.plot_perslide_roc(predictions_df, plot_results=False)

        # use subtype names
        for n in range(len(class_names)):
            auc[class_names[n]] = auc.pop(n)
        aucs['perslide'][tissue] = auc
    return aucs


def get_histocounts(sorted_list):
    histocounts = pd.read_excel(util.DATA_PATH + 'histotypes_counts_annotated.xlsx',
                                sheet_name='histotypes_counts_v3')
    histocounts = histocounts[['cancertype', 'tissue','label','sample_counts']]
    histocounts.dropna(how='any', inplace=True, axis=0)
    histocounts.columns = ['TCGA Cohort', 'Tissue', 'Histological Subtype', 'Number of Samples']

    histocounts = histocounts.dropna().groupby(['TCGA Cohort', 'Tissue', 'Histological Subtype'])['Number of Samples'].sum()
    histocounts = histocounts.reset_index().sort_values(['Tissue', 'TCGA Cohort', 'Number of Samples'])
    histocounts['Number of Samples'] = histocounts['Number of Samples'].astype(int)
    histocounts.reset_index(drop=True, inplace=True)

    histocounts = histocounts[['Tissue', 'Histological Subtype', 'Number of Samples']]

    histocounts.replace({'cervice': 'cervix',
                         'astorcytoma': 'astrocytoma',
                         'semimoma': 'seminoma',
                         'oligocytoma': 'oligoastrocytoma',
                         'non-semimoma': 'non-seminoma',
                         'adeno':'adenocarcinoma',
                         'squamous':'squamous cell',
                         'clearcell': 'clear cell'}, inplace=True)

    make_dict = lambda x: x.set_index('Histological Subtype').to_dict()['Number of Samples']
    histocounts = histocounts.groupby('Tissue')[
        ['Histological Subtype', 'Number of Samples']].apply(make_dict).to_dict()
    # histocounts['cervix'] = histocounts.pop('cervice')
    histocounts = {k:histocounts[k] for k in sorted_list}
    return histocounts


def fix_json_typos(input_json, sorted_list, drop_averages=False):
    tmp = pd.Series(input_json).apply(pd.Series).unstack().reset_index()
    if drop_averages:
        tmp = tmp[~tmp['level_0'].isin(['micro', 'macro'])]
    tmp.replace({'cervice': 'cervix',
                 'astorcytoma': 'astrocytoma',
                 'oligocytoma': 'oligoastrocytoma',
                 'semimoma': 'seminoma',
                 'non-semimoma': 'non-seminoma',
                 'adeno':'adenocarcinoma',
                 'squamous':'squamous cell',
                 'clearcell': 'clear cell',
                 'micro': 'micro-average',
                 'macro': 'macro-average'}, inplace=True)


    tmp = tmp.groupby('level_1')[['level_0', 0]].apply(lambda x: x.set_index('level_0')[0].dropna().to_dict())
    tmp = {k:tmp[k] for k in sorted_list}
    return tmp

def get_counts(counts_filename, min_sample_size = 25):
    counts = pd.read_csv(counts_filename)
    counts = counts[counts['count-method'] == 'slide']
    counts.rename(columns={'is_tumor': 'label'}, inplace=True)
    counts['label'] = counts['label'].replace({0:'normal', 1:'tumor'})
    counts['crossval_group'] = counts['crossval_group'].replace({'validation':'testing'})
    counts = counts.groupby(['crossval_group', 'label', 'cancertype'])['counts'].sum().reset_index()


    tmp = counts.groupby(['cancertype', 'label'])['counts'].sum().unstack(fill_value=0)
    cancertypes = tmp[(tmp >= min_sample_size).all(axis=1)].index
    counts = counts[counts['cancertype'].isin(cancertypes)]
    return counts

def plot_tn_counts(ax=None, log=True, counts_filename = None):
    counts = get_counts(counts_filename)

    min_sample_size = 25
    tmp = counts.groupby(['cancertype', 'label'])['counts'].sum().unstack(fill_value=0)
    cancertypes = tmp[(tmp >= min_sample_size).all(axis=1)].index
    counts = counts[counts['cancertype'].isin(cancertypes)]

    sorted_list = counts.groupby('cancertype')['counts'].sum().sort_values(ascending=False).index

    tmp1 = counts.pivot_table(index='cancertype', columns=['crossval_group' , 'label'], values='counts')
    tmp1 = tmp1.loc[sorted_list]
    tmp1.index = tmp1.index.str.upper()

    tmp1.plot.bar(stacked=True, log=log, color=['lightgreen', 'darkgreen', 'red', 'darkred'], ax=ax)

    col0 = tmp1.columns.get_level_values(0)
    col1 = tmp1.columns.get_level_values(1)
    legend = ['{:s} ({:s})'.format(y.capitalize(), x[:-3]) for x,y in zip(col0, col1)]
    ax.legend(legend, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.set_ylabel('number\nof slides', fontsize=12, fontweight='normal')
    ymin = 0.9 if log else 0
    ax.set_ylim([ymin, 2100])
    ax.set_xlabel('')
    # ax.xaxis.set_tick_params(labelsize=12)
    # ax.yaxis.set_tick_params(labelsize=12)
    return sorted_list#.tolist()
