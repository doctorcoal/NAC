import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

step_limitation = {
    'resnet_exit_quant':{
        'cifar10': 8200,
        'cifar100': 10000,
        'tiny-imagenet': 3800,
        },
    'mobilenetV2_hira_quant':{
        'cifar10': 1250,
        'cifar100': 3000,
        'tiny-imagenet': 1300,
        },
    'DeiT_tiny': {
        'cifar10': 16000,
        'cifar100': 16000,
        'tiny-imagenet': 12000
    },
}

WS = {
    'resnet_exit_quant':{
        'cifar10': 1,
        'cifar100': 5,
        'tiny-imagenet': 1,
        },
    'mobilenetV2_hira_quant':{
        'cifar10': 1,
        'cifar100': 3,
        'tiny-imagenet': 1,
        },
    'DeiT_tiny': {
        'cifar10': 5,
        'cifar100': 5,
        'tiny-imagenet': 5
    },
}

thresholds = {
    'resnet_exit_quant':{
        'cifar10': 5,
        'cifar100': -150,
        'tiny-imagenet': -150,
        },
    'mobilenetV2_hira_quant':{
        'cifar10': -150,
        'cifar100': -150,
        'tiny-imagenet': -150,
        },
    'DeiT_tiny': {
        'cifar10': -150,
        'cifar100': -150,
        'tiny-imagenet': -150
    },
}

model = ['resnet_exit_quant', 'mobilenetV2_hira_quant', 'DeiT_tiny']
dataset = ['tiny-imagenet', 'cifar100', 'cifar10']
# model = [model[2]]
# dataset = [dataset[0]]
plot_path = '.'

sns.set_context('talk')
sns.plotting_context()
sns.set_context('talk', rc={'lines.linewidth': 1.875})
plt.rcParams['font.family'] = 'Times New Roman'



for m in model:
    for d in dataset:
        score_threshold = thresholds[m][d]
        new_row = pd.DataFrame({'episode': [0], 'step': [0], 'score': [0], 'best_score': [0]})
        results = pd.read_csv(os.path.join(plot_path,m,d,"Tscores.csv"),encoding="utf-8")
        results = pd.concat([new_row, results], ignore_index=True)
        step_limit = step_limitation[m][d]
        condition = (
            (results['step'] <= step_limit) &  # 步数限制
            ((results['best_score'] > score_threshold) | (results['step'] == 0))  # 保留分数>5或初始点
        )
        results = results[condition]
        window_size = WS[m][d]
        results['smoothed'] = results['best_score'].rolling(
            window=window_size,
            center=True,
            min_periods=1  # 允许部分窗口
        ).mean()
        
        plt.figure(figsize=(10, 6), edgecolor='black')
        plt.plot(results['step'], results['smoothed'], label='Score', color=[i/256.0 for i in (178.0,36.0,0.0)], linewidth=2)
        # plt.plot(results['step'], results['best_score'], alpha=0.2, label='Score', color=[i/256.0 for i in (178.0,36.0,0.0)], linewidth=2)
        plt.tick_params(axis='y', length=0, labelsize=22)
        plt.tick_params(axis='x', length=0, labelsize=22)
        plt.xlabel('Steps', fontsize=28, fontname="Times New Roman")
        plt.ylabel('Total Score', fontsize=28, fontname="Times New Roman")
        # plt.title(model+'_'+dataset, fontsize=18, fontname="Times New Roman")
        # plt.legend(fontsize=14)
        plt.grid(True, linestyle='-.', fillstyle='left', alpha=0.9)

        # Save the plot
        if os.path.isfile(plot_path):
            os.remove(plot_path)
        plt.savefig(os.path.join(plot_path,m,d,m[:3]+'_'+d+".pdf"), dpi=300, bbox_inches="tight")
        plt.close()