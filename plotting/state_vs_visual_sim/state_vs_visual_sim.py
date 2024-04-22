# https://gist.githubusercontent.com/SamProkopchuk/b372b2b6a1b92446fa87af6b482ef460/raw/ace8c5d7af6902d66abe914ccd5e97fa63d899bb/graph.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
# save_folder = os.path.join(dir_path, "plots")
save_folder = dir_path

s = """
1 envs - train: 623.6424479484558s
1 envs - total: 631.6819670200348s
2 envs - train: 616.8938727378845s
2 envs - total: 625.05841588974s
4 envs - train: 694.7565629482269s
4 envs - total: 703.2757070064545s
32 envs - train: 707.9946808815002s
32 envs - total: 718.5912244319916s
128 envs - train: 799.6709959506989s
128 envs - total: 850.8827102184296s
"""

x = [1, 2, 4, 32, 128]
y_train = [
    623.6424479484558,
    616.8938727378845,
    694.7565629482269,
    707.9946808815002,
    799.6709959506989,
]
y_total = [
    631.6819670200348,
    625.05841588974,
    703.2757070064545,
    718.5912244319916,
    850.8827102184296,
]

sns.set_style("whitegrid")
colors = sns.color_palette()

plt.plot(x, y_train, label="visual train", linestyle="--", color=colors[0])
plt.plot(x, y_total, label="visual total", linestyle="-", color=colors[0])

s = """
1 envs - total: 654.5291013717651s
2 envs - total: 630.9965324401855s
4 envs - total: 629.2811982631683s
32 envs - total: 657.3721375465393s
128 envs - total: 731.7653551101685s
1024 envs - total: 952.9169158935547s

1 envs - train: 646.4369792938232s
2 envs - train: 622.7185454368591s
4 envs - train: 620.9312551021576s
32 envs - train: 649.061529636383s
128 envs - train: 723.0589897632599s
1024 envs - train: 940.692108631134s
"""

x = [1, 2, 4, 32, 128, 1024]
y_train = [
    646.4369792938232,
    622.7185454368591,
    620.9312551021576,
    649.061529636383,
    723.0589897632599,
    940.692108631134,
]
y_total = [
    654.5291013717651,
    630.9965324401855,
    629.2811982631683,
    657.3721375465393,
    731.7653551101685,
    952.9169158935547,
]

plt.plot(x, y_train, label="state-based train", linestyle="--", color=colors[1])
plt.plot(x, y_total, label="state-based total", linestyle="-", color=colors[1])

plt.xticks(x)
plt.xlim(left=0, right=129)

# plt.yscale("log")
# plt.xscale("log")

plt.xlabel("Number of environments")
plt.ylabel("Time (s)")
plt.legend()
filename = os.path.join(save_folder, "state_vs_visual_sim.png")
plt.savefig(filename)
print(f"Saved to {filename}")
# plt.show()