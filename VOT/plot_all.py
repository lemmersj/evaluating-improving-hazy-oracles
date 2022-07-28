"""Makes an error/DR plot at every DD constraint"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

files = os.listdir("better_smear")

run_dict = {}

# Loop through every run
for this_file in files:
    # figure out what the method being used is
    method = this_file.split("-")[0]

    # Get the data, skip if it's empty.
    with open(f"better_smear/{this_file}", "rb") as infile:
        data = pickle.load(infile)
        if data[:, -1, :].sum() == 0:
            print(f"skipping {this_file}")
            continue

    # Save/update the data in the run dict. There should be no overlap
    # between files (i.e., all sums should be float+0.0)
    if method not in run_dict:
        run_dict[method] = data
    else:
        run_dict[method] = run_dict[method]+data

x = np.array([*range(101)])/101
# Loop through every depth
for depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    # Then loop through every method (guess this could be a key iteration...)
    for method in ['naive','smart','ensemble_mean','combined']:
        if "naive" in method:
            color = "#FE6100"
            label = "Naive Replacement"
        elif "smart" in method:
            color = "#DC267F"
            label = "Smart Replacement"
        elif "combined" in method:
            color = "#785EF0"
            label = "Ours"
        elif "mean" in method:
            label = "Mean"
            color = "#648FFF"
        # calc mean and stderr, then plot
        this_run = run_dict[method][depth-1, :, :].mean(axis=1)
        this_run_stderr = run_dict[method][depth-1, :, :].std(axis=1)/np.sqrt(this_run.shape[0])
        plt.plot(x, this_run, color=color, label=label)
        plt.fill_between(x, this_run-this_run_stderr,
                         this_run+this_run_stderr, color=color, alpha=0.5)
    plt.xlabel("Deferral Rate")
    plt.ylabel("Mean Error (1-IoU)")
    plt.title(f"Depth: {depth}")
    plt.legend()
    plt.savefig(f"single_rqd_plots/{depth}.pdf")
    plt.clf()
