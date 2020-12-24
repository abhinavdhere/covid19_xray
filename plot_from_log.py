import os
import re
import sys
import matplotlib.pyplot as plt


def get_values(process, metric_name, all_text):
    all_lines = re.findall(r'.*'+process+r'.*', all_text)
    metric_values = []
    for line in all_lines:
        value = re.findall(
            metric_name+r'\s:\s[0-9.]+', line)[0].split(':')[1].strip(' ')
        metric_values.append(float(value))
    return metric_values


if __name__ == '__main__':
    save_name = sys.argv[1]
    metric_name = sys.argv[2]
    plot_types = sys.argv[3]
    colors = ['g', 'm', 'r']
    all_text = open(os.path.join('logs', 'log_'+save_name+'.txt'), 'r').read()
    for idx, process in enumerate(plot_types.split('-')):
        metric_values = get_values(process, metric_name, all_text)
        plt.plot(list(range(1, len(metric_values)+1)), metric_values,
                 colors[idx])
    plt.ylabel(metric_name)
    plt.xlabel('Epochs')
    plt.title('Performance over epochs')
    plt.gca().legend(plot_types.split('-'))
    plt.savefig('plots/'+save_name+'_'+metric_name+'.png')
