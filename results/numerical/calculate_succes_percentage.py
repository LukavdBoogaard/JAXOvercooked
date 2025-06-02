import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import json
import glob
import numpy as np
from typing import Sequence
from pathlib import Path
import statistics
import matplotlib.pyplot as plt

def chunk_list_by(array: list[float], seq_length) -> list[list[float]]:
    n = len(array) // seq_length
    return [array[i * n:(i + 1) * n] for i in range(seq_length)]

def calculate_avg_solved(base_path, file_name, seed, seq_len):
    full_path = base_path / f"seed_{seed}" / file_name

    with full_path.open("r") as f:
        data = json.load(f)
    
    data = chunk_list_by(data, seq_len)

    means = [statistics.mean(seq) for seq in data]
    count = len([mean for mean in means if mean > 10])
    return count, means

def loop_over_seeds(base, seq_len, seeds, file_name):
    avg_count = []
    total = 0
    for seed in seeds:
        count, means = calculate_avg_solved(base, file_name, seed, seq_len)
        avg_count.append(count)
        total += count
    
    avg_count = total/len(avg_count)
    return avg_count

ippo_path = Path("/home/luka/repo/JAXOvercooked/results/data/experiment_1/ippo/CNN/random_5")
file_name = "training_reward.json"
avg_count_ippo = loop_over_seeds(ippo_path, 5, [0, 1, 2, 3, 4], file_name)
vdn_path = Path("/home/luka/repo/JAXOvercooked/results/data/experiment_1/vdn/CNN/random_5")
avg_count_vdn = loop_over_seeds(vdn_path, 5, [0, 1, 2, 3, 4], file_name)

ippo_path = Path("/home/luka/repo/JAXOvercooked/results/data/experiment_1/ippo/CNN/random_15")
file_name = "training_reward.json"
avg_count_ippo_15 = loop_over_seeds(ippo_path, 15, [0, 2, 3, 4, 5], file_name)
vdn_path = Path("/home/luka/repo/JAXOvercooked/results/data/experiment_1/vdn/CNN/random_15")
avg_count_vdn_15 = loop_over_seeds(vdn_path, 15, [0, 1, 2, 3, 4], file_name)

print(avg_count_ippo)
ippo_5_percentage = avg_count_ippo / 5
vdn_5_percentage = avg_count_vdn / 5

ippo_15_percentage = avg_count_ippo_15 / 15
vdn_15_percentage = avg_count_vdn_15 / 15

labels = ['5', '15']  # Number of environments/tasks/whatever '5' and '15' mean
ippo = [ippo_5_percentage, ippo_15_percentage]
vdn = [vdn_5_percentage, vdn_15_percentage]

x = np.arange(len(labels))  # label locations
width = 0.30  # width of the bar


fig, ax = plt.subplots(figsize=(6, 4.5))
rects1 = ax.bar(x - width/2, ippo, width, label='IPPO')
rects2 = ax.bar(x + width/2, vdn, width, label='VDN')

# Add some text for labels, title and axes ticks
ax.set_ylabel('Success Percentage')
ax.set_xlabel('Sequence Length')
# ax.set_title('Comparison of IPPO and VDN')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Optionally, add labels to the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("/home/luka/repo/JAXOvercooked/results/plots/ippo_vdn_comparison.png")
plt.show()




    
    
   