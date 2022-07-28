"""Produces a mean error/DR plot for every DD constraint.

found in the supplemental material.

Typical usage:
    python build_big_csv.py armae_arrays/compiled.pickle
"""
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

#plt.rc('axes',titlesize=24)
#plt.rc('axes',labelsize=22)
#plt.rc('xtick',labelsize=18)
#plt.rc('ytick',labelsize=18)
#plt.rc('legend',fontsize=18)
pickle_file = sys.argv[1]

method_to_label = {'naive':'Naive Replacement', 'smart':'Smart Replacement',
                   'ensemble_consensus':'Ensemble Consensus',
                   'ensemble_mean': 'Ensemble Mean', 'combined': 'Ours'}
method_to_color = {'naive':'#FE6100', 'smart':'#DC267F',
                   'ensemble_consensus':'#FFB000', 'ensemble_mean': '#648FFF',
                   'combined': "#785EF0"}

# Open the pickle
with open(pickle_file, "rb") as infile:
    data = pickle.load(infile)

# Iterate through splits and depth constraints, each of which creates
# its own plot.
for split in ["val","testA","testB"]:
    for depth in [1,2,3,4,5,6,7,8,9,10]:
        # Now iterate through methods, which are lines in the plot.
        for method in ['naive','smart','ensemble_consensus','ensemble_mean','combined']:
            # Pull the correct data from the pickle.
            cur_key = f"{split}-UNITER-gt-3-entropy-{method}_{depth}"
            cur_data = data[cur_key]['errors']
            # Calculate the mean and standard error at every DR
            cur_marginal = cur_data.mean(axis=0)
            cur_marginal_stderr = cur_data.std(axis=0)/np.sqrt(cur_data.shape[0])

            # and plot
            x = np.arange(cur_marginal.shape[0])/cur_marginal.shape[0]
            plt.plot(x, cur_marginal, label=method_to_label[method],\
                     color=method_to_color[method])
            plt.fill_between(x, cur_marginal-cur_marginal_stderr,\
                             cur_marginal+cur_marginal_stderr,\
                             color=method_to_color[method], alpha=0.5)
        # save, then move on to the next split/depth combo.
        plt.xlabel("Re-Query Rate")
        plt.ylabel("Mean Error")
        plt.title(f"{split} Split, RQD: {depth}")
        plt.legend()
        plt.savefig(f"single_rqd_plots/{split}-{depth}.pdf")
        plt.clf()
