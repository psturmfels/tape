import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Figures for accuracy/time')
    parser.add_argument('csv_file', type=str, help='Where the time_structure_pred results are')
    parser.add_argument('--output_dir', type=str, default='/export/home/tape/profile_depths/figures/')
    return parser

def plot_time(df):
    fig = plt.figure(dpi=150)
    ax = fig.gca()
    x_inds = np.arange(df.shape[0])
    x_ticks = list(df['model_type'])
    x_ticks = ['Profile Prediction' if tick == 'transformer' else 'Netsurfp2.0' for tick in x_ticks]
    x_ticks = [tick + f' ({db})' if tick =='Netsurfp2.0' else tick for tick, db in zip(x_ticks, df['hhdb'])]

    search = ax.bar(x_inds, df['search_per_sequence'], 0.8)
    pred   = ax.bar(x_inds, df['pred_per_sequence'], 0.8)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(0.1)
    ax.spines['right'].set_linewidth(0.1)
    ax.set_title('Computation Time of Secondary Structure Predictors', fontsize=14)
    ax.set_ylabel('Computation Time (seconds) Per Sequence')
    ax.set_xticks(x_inds)
    ax.set_xticklabels(x_ticks)
    ax.legend((search[0], pred[0]), ('Search Time', 'Prediction Time'))
    return fig, ax

def plot_acc(df):
    fig = plt.figure(dpi=150)
    ax = fig.gca()
    x_inds = np.arange(df.shape[0])
    x_ticks = list(df['model_type'])
    x_ticks = ['Profile Prediction' if tick == 'transformer' else 'Netsurfp2.0' for tick in x_ticks]
    x_ticks = [tick + f' ({db})' if tick =='Netsurfp2.0' else tick for tick, db in zip(x_ticks, df['hhdb'])]

    _ = ax.bar(x_inds, df['accuracy'], 0.8)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(0.1)
    ax.spines['right'].set_linewidth(0.1)
    ax.set_title('Accuracy of Secondary Structure Predictors', fontsize=14)
    ax.set_ylabel('Mean Accuracy')
    ax.set_xticks(x_inds)
    ax.set_xticklabels(x_ticks)
    return fig, ax

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()
    df = pd.read_csv(args.csv_file)

    fig, ax = plot_time(df)
    fig.savefig(os.path.join(args.output_dir, 'ss_time_comp.png'))

    fig, ax = plot_acc(df)
    fig.savefig(os.path.join(args.output_dir, 'ss_acc_comp.png'))

if __name__ == '__main__':
    main()
