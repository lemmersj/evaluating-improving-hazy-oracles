"""Uses the compiled pickle to generate a CSV of DEV metrics, and acc@1.

Tables 2 and 3 at the time of this writing.

Typical usage:
    python calc_dev.py armae_arrays/compiled.pickle
"""
import pickle
import sys
import numpy as np

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

ares_and_keys = {}

for method in with_depth_dict:
    method_areas_under = []
    method_accs_1 = []
    for trial in range(100):
        current_areas_under = []
        current_accs_1 = []
        for depth_constraint in with_depth_dict[method]:
            if depth_constraint > 10 or depth_constraint <= 0:
                continue
            row = np.random.randint(with_depth_dict[method][depth_constraint]['errors'].shape[0])
            current_areas_under.append(
                with_depth_dict[method][depth_constraint]\
                ['errors'][row, :].mean())
            current_accs_1.append(
                with_depth_dict[method][depth_constraint]\
                ['errors'][row, -1].mean())
        method_areas_under.append(np.array(current_areas_under).mean())
        method_accs_1.append(np.array(current_accs_1).mean())

    ares_and_keys[method] = {}
    ares_and_keys[method]['mean'] = (np.array(method_areas_under)*100).mean()
    ares_and_keys[method]['stderr'] = (np.array(
        method_areas_under)*100).std()/np.sqrt(len(method_areas_under))
    ares_and_keys[method]['accs_1'] = (np.array(method_accs_1)*100).mean()
    ares_and_keys[method]['stderr_accs_1'] = (
        np.array(method_accs_1)*100).std()/np.sqrt(len(method_accs_1))

with open("performance.csv", "w") as csv_file:
    csv_file.write("split,net,source,dist,score,selection,AU,stderr,err@1,stderr\n")
    for key in ares_and_keys:
        # A manual split of the keys back to semantic information
        split_key = key.split("-")
        split = split_key[0]
        net = split_key[1]
        source = split_key[2]
        dist = dist_dict[int(split_key[3])]
        score = split_key[4]
        selection = split_key[5]
        csv_file.write(f'{split},{net},{source},{dist},{score},{selection},'\
                       f'{ares_and_keys[key]["mean"]},{ares_and_keys[key]["stderr"]},'
                       f'{ares_and_keys[key]["accs_1"]},{ares_and_keys[key]["stderr_accs_1"]}\n')
