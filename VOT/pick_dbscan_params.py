"""Runs a gridsearch to pick dbscan params"""
import os
import csv
import numpy as np

directories = os.listdir("output_videos_wh_correlated")

grid_ys = [1, 3, 5, 10, 15, 20, 50]
grid_xes = [3, 5, 10, 15, 20]

# Save all the errors in a grid
vals = np.zeros((7,5))

best = 0
# Loop through all directories (corresponding to a dbscan trial).
for directory in directories:
    if not os.path.exists(f"output_videos_wh_correlated/{directory}/part_vs_det/summary.csv"):
        print(f"skipping {directory}")
        continue
    if "True" not in directory:
        continue

    # Figure out what the parameters are, and map them to the grid
    split_dir = directory.split("-")
    eps = int(float(split_dir[0]))
    y_loc = np.where(np.array(grid_ys) == eps)[0][0]
    cluster = int(split_dir[1])
    x_loc = np.where(np.array(grid_xes) == cluster)[0][0]
    stoch_ious = []
    det_ious = []

    # Pull the IoU from the csv.
    with open(f"output_videos_wh_correlated/{directory}/part_vs_det/summary.csv") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            stoch_ious.append(float(row['iou-stoch']))
            det_ious.append(float(row['iou-det']))

    # Figure out what the mean IoU is
    stoch_mean = np.array(stoch_ious).mean()
    # and save it if it's the best
    if stoch_mean > best:
        best = stoch_mean
        best_file = directory

    # convert to error and save to the grid
    vals[y_loc, x_loc] = 1-stoch_mean

    # report the result
    print(f"{directory}: {stoch_mean} vs {np.array(det_ious).mean()}")

# round the values and write them to the csv
vals = np.around(vals*10000)/10000
with open("dbscan_results.csv", "w") as in_csv:
    in_csv.write(",3,5,10,15,20\n")
    for row in range(len(vals)):
        in_csv.write(f"{grid_ys[row]},")
        for val in vals[row]:
            in_csv.write(f"{val},")
        in_csv.write("\n")

# report the best result.
print(f"Best is {best} at {best_file}")
