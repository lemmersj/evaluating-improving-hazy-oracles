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

#plt.rc('axes',titlesize=24)
#plt.rc('axes',labelsize=22)
#plt.rc('xtick',labelsize=18)
#plt.rc('ytick',labelsize=18)
#plt.rc('legend',fontsize=18)
pickle_file = sys.argv[1]

method_to_label = {'naive':'Naive Replacement', 'smart':'Smart Replacement','ensemble_consensus':'Consensus', 'ensemble_mean': 'Mean', 'combined': 'Ours'}
method_to_color = {'naive':'#FE6100', 'smart':'#DC267F','ensemble_consensus':'#FFB000', 'ensemble_mean': '#648FFF', 'combined': "#785EF0"}

# I just did this manually here to save queries to the database
dist_dict = {1: 'softmax', 2: 'varratio', 3:'dropout', 4:'dropout_textonly',
             5:'varratio_textonly'}

# Open the pickle
with open(pickle_file, "rb") as infile:
    data = pickle.load(infile)

for split in ["val","testA","testB"]:
    for depth in [1,2,3,4,5,6,7,8,9,10]:
        for method in ['naive','smart','ensemble_consensus','ensemble_mean','combined']:
            cur_key = f"{split}-UNITER-gt-3-entropy-{method}_{depth}"
            cur_data = data[cur_key]['errors']
            cur_marginal = cur_data.mean(axis=0)
            cur_marginal_stderr = cur_data.std(axis=0)/np.sqrt(cur_data.shape[0])
            cur_marginal = cur_marginal*100
            cur_marginal_stderr = cur_marginal_stderr * 100

            x = np.arange(cur_marginal.shape[0])/cur_marginal.shape[0]
            plt.plot(x, cur_marginal, label=method_to_label[method], color=method_to_color[method])
            plt.fill_between(x, cur_marginal-cur_marginal_stderr, cur_marginal+cur_marginal_stderr, color=method_to_color[method], alpha=0.5)
        plt.xlabel("Deferral Rate")
        plt.ylabel("Mean Error (%)")
        plt.title(f"Split: {split}, DDC: {depth}")
        plt.legend()
        plt.savefig(f"single_rqd_plots/{split}-{depth}.pdf")
        plt.clf()
