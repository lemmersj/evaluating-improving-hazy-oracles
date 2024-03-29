"""Uses the compiled pickle to generate a CSV of ARE metrics.

Tables 2 and 5 at the time of this writing.

Typical usage:
    python build_big_csv.py armae_arrays/compiled.pickle
"""
import pickle
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt

plt.rc('axes',titlesize=24)
plt.rc('axes',labelsize=22)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('legend',fontsize=18)
pickle_file = sys.argv[1]

# I just did this manually here to save queries to the database
dist_dict = {1: 'softmax', 2: 'varratio', 3:'dropout', 4:'dropout_textonly',
             5:'varratio_textonly'}

# Open the pickle
with open(pickle_file, "rb") as infile:
    data = pickle.load(infile)

# 3D integration.
# Dict of dicts. First key is the actual method, second key is RQ depth.
with_depth_dict = {}
for key in data:
    split_key = key.split("_")
    if len(split_key) == 3:
        split_key = ["_".join([split_key[0], split_key[1]]), split_key[2]]
    method = split_key[0]
    depth_constraint = int(split_key[1])

    if method not in with_depth_dict.keys():
        with_depth_dict[method] = {}

    with_depth_dict[method][depth_constraint] = data[key]

depth_marginals = {}
for method in with_depth_dict:
    method_areas_under = []
    for trial in range(100):
        current_areas_under = []
        for depth_constraint in range(1, 11):
            row = np.random.randint(with_depth_dict[method][depth_constraint]['errors'].shape[0])
            current_areas_under.append(with_depth_dict[method][depth_constraint]['errors'][row, :].mean())

        method_areas_under.append(np.array(current_areas_under))

    depth_marginals[method] = {}
    depth_marginals[method]['mean'] = (np.array(method_areas_under)*100).mean(axis=0)
    depth_marginals[method]['stderr'] = (np.array(method_areas_under)*100).std(axis=0)/np.sqrt(len(method_areas_under))

# rqr_marginals
rqr_marginals = {}
for method in with_depth_dict:
    method_areas_under = []
    for trial in range(100):
        current_areas_under = []
        for depth_constraint in range(1, 11):
            row = np.random.randint(with_depth_dict[method][depth_constraint]['errors'].shape[0])
            current_areas_under.append(with_depth_dict[method][depth_constraint]['errors'][row, :])

        method_areas_under.append(np.array(current_areas_under).mean(axis=0))

    rqr_marginals[method] = {}
    rqr_marginals[method]['mean'] = (np.array(method_areas_under)*100).mean(axis=0)
    rqr_marginals[method]['stderr'] = (np.array(method_areas_under)*100).std(axis=0)/np.sqrt(len(method_areas_under))

x = [*range(1, 11)]
split = "testB"
fig, (ax1, ax2) = plt.subplots(1, 2)
for this_key in depth_marginals.keys():
    if split not in this_key:
        continue
    if "3" not in this_key:
        continue
    legend_name = this_key.split("_")[-1].split("-")[-1]
    if "consensus" in legend_name:
        color = "#FFB000"
    elif "naive" in legend_name:
        color = "#FE6100"
    elif "smart" in legend_name:
        color = "#DC267F"
    elif "combined" in legend_name:
        color = "#785EF0"
    elif "mean" in legend_name:
        color = "#648FFF"

    ax1.plot(x, depth_marginals[this_key]['mean'], label=legend_name, marker="o", color=color)
    ax1.fill_between(x,
        depth_marginals[this_key]['mean']-depth_marginals[this_key]['stderr'], depth_marginals[this_key]['mean']+depth_marginals[this_key]['stderr'], alpha=0.5, color=color)
    ax1.set_xticks(x)

for this_key in depth_marginals.keys():
    if split not in this_key:
        continue
    if "3" not in this_key:
        continue
    legend_name = this_key.split("_")[-1].split("-")[-1]
    if "consensus" in legend_name:
        color = "#FFB000"
    elif "naive" in legend_name:
        color = "#FE6100"
    elif "smart" in legend_name:
        color = "#DC267F"
    elif "combined" in legend_name:
        color = "#785EF0"
    elif "mean" in legend_name:
        color = "#648FFF"
    x = np.array([*range(len(rqr_marginals[this_key]['mean']))])/len(rqr_marginals[this_key]['mean'])
    ax2.plot(x, rqr_marginals[this_key]['mean'], label=legend_name, linewidth=0.5, color=color)
    ax2.fill_between(x,
        rqr_marginals[this_key]['mean']-rqr_marginals[this_key]['stderr'], rqr_marginals[this_key]['mean']+rqr_marginals[this_key]['stderr'], alpha=0.5, color=color)

ax1_ylim = ax1.get_ylim()
ax2_ylim = ax2.get_ylim()
ylim_min = min(ax1_ylim[0], ax2_ylim[0])
ylim_max = max(ax1_ylim[1], ax2_ylim[1])

ax1.set_ylim([ylim_min, ylim_max])
ax2.set_ylim([ylim_min, ylim_max])

ax2.axes.yaxis.set_visible(False)

ax1.set_xlabel("Deferral Depth Constraint")
ax1.set_ylabel("Mean Error (%)")
ax2.set_xlabel("Deferral Rate")
ax2.legend(loc=(1.1, 0.5))
fig.set_size_inches(20, 6)
plt.tight_layout()
plt.savefig("marginals.pdf")

plt.clf()
x = [*range(1, 11)]
for split in ["val", "testA", "testB"]:
    for this_key in depth_marginals.keys():
        if split not in this_key:
            continue
        if "3" not in this_key:
            continue
        plt.plot(x, depth_marginals[this_key]['mean'], label=this_key, marker="o")
        plt.fill_between(x, 
            depth_marginals[this_key]['mean']-depth_marginals[this_key]['stderr'], depth_marginals[this_key]['mean']+depth_marginals[this_key]['stderr'], alpha=0.5)
    plt.legend()
    plt.savefig(f"depth_marginal_{split}.pdf")
    plt.clf()

for split in ["val", "testA", "testB"]:
    for this_key in depth_marginals.keys():
        if split not in this_key:
            continue
        if "3" not in this_key:
            continue
        x = np.array([*range(len(rqr_marginals[this_key]['mean']))])/len(rqr_marginals[this_key]['mean'])
        plt.plot(x, rqr_marginals[this_key]['mean'], label=this_key)
        plt.fill_between(x,
            rqr_marginals[this_key]['mean']-rqr_marginals[this_key]['stderr'], rqr_marginals[this_key]['mean']+rqr_marginals[this_key]['stderr'], alpha=0.5)
    plt.legend()
    plt.savefig(f"dr_marginal_{split}.pdf")
    plt.clf()

