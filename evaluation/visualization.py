import os
import html

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pickle

from IPython.core.display import display, HTML

from utils.text_processing import text_tokenizer, text_to_input, input_to_text

pd.set_option('display.max_colwidth', None)

CHECKPOINT_PATH = './checkpoints'

def weights_plot(text, weights):
    tick_locs = np.arange(0, len(text), dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.bar(tick_locs, weights)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(text)

    plt.plot()


def text_highlighter(text, weights, max_alpha=0.75):
    """Colours the strings according to some given weights.

    Args:
        text ([iterable str])
        weights ([iterable float])
        max_alpha (float, optional): The maximum darkness of a colour. Defaults to 0.7.

    Returns:
        HTML encoded string with colours.
    """
    # https://stackoverflow.com/questions/59220488/to-visualize-attention-color-tokens-using-attention-weights

    if isinstance(text, str):
        text = text_tokenizer(text)

    cmap = matplotlib.cm.Blues
    template = '<span class="barcode"; style="color: {}; background-color: {}">{}</span>'

    weights = max_alpha * np.array(weights) / max(weights)

    colored_string = []
    for word, weight in zip(text, weights[1:-1]):
        color = matplotlib.colors.rgb2hex(cmap(weight)[:3])
        colored_string.append(template.format('black' if weight < 0.85 else 'white', color,
                                              '&nbsp' + html.escape(word) + '&nbsp'))

    return HTML(' '.join(colored_string))

def results_loader(version, evaluation='snli', config='default', agg_method='macro'):
    """Loads in the evaluation results and makes a pretty table out of it.

    Args:
        version (int): Which model version to load in.
        evaluation (str, optional): Evaluation profile. Choose from 'snli' (default), 'infersent', 'probing_all', 'coco'. Last 3 match SentEval tasks.
        config (str, optional): InferSent classifier profile. Most likely 'default'.
        agg_method (str, optional): Method for aggregating over subsets (where relevant). Options are 'macro' or 'micro'.
    """

    if evaluation == 'snli':
        eval_file = "snli_eval.pkl"
    else:
        eval_file = "SentEval_" + config + "_" + evaluation + ".pkl"

    evals = dict()
    models = [model for model in os.listdir(CHECKPOINT_PATH) if str(version) in model]
    for model in models:
        if eval_file in os.listdir(os.path.join(CHECKPOINT_PATH, model)):
            with open(os.path.join(CHECKPOINT_PATH, model, eval_file), "rb+") as file:
                model_perf = pickle.load(file)

            encoder_name = model.rsplit("-")[1].rsplit("_")[0]
            evals[encoder_name] = model_perf

    if evaluation == 'snli':
        evals_df = snli_printer(evals)
    elif evaluation == 'infersent':
        evals_df = senteval_printer(evals, agg_method)
    elif evaluation == 'probing_all':
        evals_df = probing_printer(evals)
    else:
        raise ValueError("Unknown evaluation type")

    return evals_df

def snli_printer(evals):

    task_order = ['Train', 'Valid', 'Test']
    model_order = ['Baseline', 'Simple', 'BiSimple', 'BiMaxPool']
    evals_df = []
    for encoder in evals.keys():
        for dataset in evals[encoder].keys():
            value = evals[encoder][dataset]

            value = "{:4.1f}".format(value * 100)

            evals_df.append((encoder, dataset, value))

    evals_df = pd.DataFrame.from_records(evals_df, columns=["Model", "Task", "Score"])
    evals_df = evals_df.pivot(index="Model", columns="Task")
    evals_df.columns = evals_df.columns.droplevel()
    evals_df = evals_df.reindex(columns=task_order)
    evals_df = evals_df.reindex(model_order)
    evals_df = evals_df.reindex(model_order)

    conneau_results = pd.DataFrame([['-', '85.0', '84.5']],
                                columns=task_order,
                                index=["BiLSTM-Max (SNLI)*"])

    evals_df = evals_df.append(conneau_results)
    evals_df.index.name = None

    return evals_df

def senteval_printer(evals, agg_method):

    task_order = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICK-R', 'SICK-E', 'STS14']
    model_order = ['Baseline', 'Simple', 'BiSimple', 'BiMaxPool']
    evals_df = []
    for encoder in evals.keys():
        for dataset in evals[encoder].keys():

            log = evals[encoder][dataset]

            if 'STS' in dataset:
                if agg_method == 'macro':
                    pearson = "{:4.2f}".format(log['all']['pearson']['mean'])
                    spearman = "{:4.2f}".format(log['all']['spearman']['mean'])
                else:
                    pearson = "{:4.2f}".format(log['all']['pearson']['wmean'])
                    spearman = "{:4.2f}".format(log['all']['spearman']['wmean'])
                value = pearson.lstrip('0') + "/" + spearman.lstrip('0')

                evals_df.append((encoder, dataset, value))

            elif 'SICKRelatedness' in dataset:
                if agg_method == 'macro':
                    pearson = (log['devpearson'] + log['pearson']) / 2
                else:
                    pearson = (log['devpearson'] * log['ndev'] + log['pearson'] *
                               log['ntest']) / (log['ndev'] + log['ntest'])

                pearson = "{:4.3f}".format(pearson)
                evals_df.append((encoder, dataset[:4] + "-" + dataset[4], pearson))

            else:
                if agg_method == 'macro':
                    acc = (log['devacc'] + log['acc']) / 2
                else:
                    acc = (log['devacc'] * log['ndev'] + log['acc'] * log['ntest']) / (log['ndev'] + log['ntest'])

                if 'MRPC' in dataset:
                    acc = "{:4.1f}".format(acc)
                    fscore = "{:4.1f}".format(log['f1'])
                    value = acc + "/" + fscore
                else:
                    value = "{:4.1f}".format(acc)

                if 'SICK' in dataset:
                    evals_df.append((encoder, dataset[:4] + "-" + dataset[4], value))
                else:
                    evals_df.append((encoder, dataset, value))

    evals_df = pd.DataFrame.from_records(evals_df, columns=["Model", "Task", "Score"])
    evals_df = evals_df.pivot(index="Model", columns="Task")
    evals_df.columns = evals_df.columns.droplevel()
    evals_df = evals_df.reindex(columns=task_order)
    evals_df = evals_df.reindex(model_order)

    conneau_results = pd.DataFrame([['79.9', '84.6', '92.1', '89.8', '83.3',
                                    '88.7', '75.1/82.3', '0.885', '86.3', '.68/.65']],
                                   columns=task_order,
                                   index=["BiLSTM-Max (SNLI)*"])

    evals_df = evals_df.append(conneau_results)
    evals_df.index.name = 'Macro Avg.' if agg_method == 'macro' else 'Micro Avg.'

    return evals_df

def probing_printer(evals):

    task_order = ['Length', 'WordContent', 'Depth', 'TopConstituents',
                  'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    model_order = ['Baseline', 'Simple', 'BiSimple', 'BiMaxPool']
    evals_df = []
    for encoder in evals.keys():
        for dataset in evals[encoder].keys():
            log = evals[encoder][dataset]

            acc = (log['devacc'] + log['acc']) / 2

            value = "{:4.1f}".format(acc)

            evals_df.append((encoder, dataset, value))

    evals_df = pd.DataFrame.from_records(evals_df, columns=["Model", "Task", "Score"])
    evals_df = evals_df.pivot(index="Model", columns="Task")
    evals_df.columns = evals_df.columns.droplevel()
    evals_df = evals_df.reindex(columns=task_order)
    evals_df = evals_df.reindex(model_order)

    conneau_results = pd.DataFrame([['71.7', '87.3', '41.6', '70.5', '65.1', '86.7', ' 80.7', '80.3', '62.1', '66.8'],
                                    ['100 ', '100 ', '84.0', '84.0', '98.0', '85.0', '88.0', '86.5', '81.2', '85.0']],
                                   columns=task_order,
                                   index=["BiLSTM-Max (NLI)*", "Human*"])

    evals_df = evals_df.append(conneau_results)

    task_cat = ['Surface', 'Surface', 'Syntax', 'Syntax', 'Syntax',
                'Semantics', 'Semantics', 'Semantics', 'Semantics', 'Semantics']

    evals_df.columns = pd.MultiIndex.from_tuples(list(zip(task_cat, task_order)))
    evals_df = evals_df.rename(columns={'Length':'SentLen', 'WordContent':'WC', 'Depth':'TreeDepth', 'TopConstituents':'TopConst',
                                        'BigramShift':'BShift', 'Tense':'Tense', 'SubjNumber':'SubjNum', 'ObjNumber':'ObjNum', 'OddManOut':'SOMO', 'CoordinationInversion':'CoordInv'})

    return evals_df

def linguistic_properties_printer(version, property, N_lim=100, figsize=(10, 10)):
    """Prints the linguistic property summary files.

    Args:
        version: model version
        property: linguistic property to display. Choice of 'pos', 'tag', 'dep', 'iob'.
        N_lim (int, optional): minimal number of observations for inclusion. Defaults to 100.
        figsize (tuple, optional): defaults to (10, 10).
    """

    fp = f"C:/Users/ivoon/Documents/GitHub/ATCS_Practical1/checkpoints/InferSent-BiMaxPool_v{str(version)}/linguistic_properties_summary.pkl"
    if os.path.exists(fp):
        with open(fp, 'rb+') as file:
            linguistic_properties_summary = pickle.load(file)
    else:
        print("File not found.")
        print(fp)
        return

    cmap = matplotlib.cm.viridis_r

    boxes = []
    sorted_summary = {k: v for k, v in sorted(
        linguistic_properties_summary[property].items(), key=lambda item: item[1]['mean'])}
    for k in sorted_summary.keys():
        summary_stats = sorted_summary[k]
        if summary_stats['N'] <= N_lim:
            continue
        SE = np.sqrt(summary_stats['std']) / np.sqrt(summary_stats['N'])
        box = {'label': k,
               'mean': summary_stats['mean'],
               'whislo': summary_stats['quantiles'][1],
               'q1': summary_stats['quantiles'][2],
               'med': summary_stats['quantiles'][3],
               'q3': summary_stats['quantiles'][4],
               'whishi': summary_stats['quantiles'][5],
               'fliers': []
               }
        boxes.append(box)

    boxprops = dict(linewidth=2, edgecolor='black', facecolor='white')
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    meanprops = dict(marker='d', markeredgecolor='black', markerfacecolor='white')

    fig, ax = plt.subplots(figsize=figsize)
    bplot = ax.bxp(boxes, patch_artist=True,
                   showfliers=True, vert=False,
                   showmeans=True, showcaps=False,
                   boxprops=boxprops,
                   medianprops=medianprops,
                   meanprops=meanprops)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(labelsize=12)
    ax.set_ylabel("Property", fontsize=14)
    ax.set_xlabel("Selection Propensity over Random", fontsize=14)

    ax.xaxis.set_ticks_position('none')
    ax.set_xlim(0)

    ax.set_title('BiLSTM-Max Selection Propensity', fontsize=16)

    colors = ['pink', 'lightblue', 'lightgreen']
    for i, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(cmap(i / len(sorted_summary.keys())))

def input_output_table(premise, hypothesis, label, model, vocab, loss=None, N=50):
    """Generates a human understandable IO table for the models.

    Args:
        premise (LongTensor): processed premise (e.g. from dataloader)
        hypothesis (LongTensor): processed hypothesis (e.g. from dataloader)
        label (LongTensor or list of ints): iterable of labels corresponding to examples
        model (torch.nn module or pl.module): model for inference
        loss: if None, calculates, else print given loss
    """
    def _table_cmap(s):

        cmap = plt.get_cmap('Blues')

        return ['background-color: {:}'.format(matplotlib.colors.rgb2hex(cmap(0.75 * val))) for val in s]

    premise_text = input_to_text(premise, vocab)
    hypothesis_text = input_to_text(hypothesis, vocab)

    logits = model.forward(premise, hypothesis)
    if loss == None:
        loss = F.cross_entropy(logits, torch.LongTensor(label), reduction='none').tolist()

    if isinstance(label, torch.Tensor):
        label = label.tolist()

    table = pd.DataFrame.from_records(list(zip(premise_text,
                                               hypothesis_text,
                                               label,
                                               *F.softmax(logits, dim=-1).T.tolist(),
                                               loss
                                               )),
                                      columns=['Premise', 'Hypothesis', 'Label', 'p(f(x)=0)', 'p(f(x)=1)', 'p(f(x)=2)', 'Loss'])

    table = table[:N].style.apply(_table_cmap,
                              subset=['p(f(x)=0)', 'p(f(x)=1)', 'p(f(x)=2)'])

    return table
