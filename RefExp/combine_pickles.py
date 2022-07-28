"""Combines pickles generated in parallel to a single pickle.

Typical usage example:
    python combine_pickles.py armae_arrays
"""

import pickle
import sys
import os
import numpy as np

directory = sys.argv[1]

files = os.listdir(directory)

# The dict that gets built then saved.
compiled_dict = {}
# Loop through all the files in the directory

for filename in files:
    # Don't try to put compiled.pickle in itself.
    if "compiled" in filename:
        continue

    # And ignore any other filetypes
    if filename.split(".")[-1].lower() != "pickle":
        continue

    # Open the input pickle
    with open(os.path.join(directory, filename), "rb") as in_file:
        cur_dict = pickle.load(in_file)

    # Accidentally messed up one of the dict keys in generate_performance_pickles
    # Fortunately, completly fixable (but ugly)
    new_dict = {}
    for key in cur_dict.keys():
        changed_key = key
        if "{" in key:
            method = filename.split("-")[1]
            changed_key = changed_key.replace("{replacement_method}", method)
        new_dict[changed_key] = cur_dict[key]
    cur_dict = new_dict
    # Within the current pickle...
    for key in cur_dict:
        # ... update the compiled pickle
        compiled_dict[key] = cur_dict[key]
        if np.where(cur_dict[key]['rmaes'].sum(axis=0) == 0)[0].shape[0] > 0:
            # for some of our methods, we don't update every coverage.
            # e.g., because we need 3 queries to decide. So in cases where
            # the value is zero, set it to the previous (higher coverage) val
            for i in range(cur_dict[key]['rmaes'].shape[1]):
                if cur_dict[key]['rmaes'][:,i].sum() == 0:
                    cur_dict[key]['rmaes'][:,i] = cur_dict[key]['rmaes'][:, i-1]
                    cur_dict[key]['errors'][:,i] = cur_dict[key]['errors'][:, i-1]

# save
with open(os.path.join(directory, "compiled.pickle"), "wb") as out_file:
    pickle.dump(compiled_dict, out_file)
