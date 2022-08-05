"""Calculates the deferral-free and perfect rejection error of the VOT tracker."""
import csv
import random
import numpy as np

# Begin by building a dictionary, where the IoUs for every vid/annotator are saved.
annot_dict = {}
with open ("output_videos_wh_correlated/10.0-20-True/part_vs_det/summary.csv", "r") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        if row['video'] not in annot_dict:
            annot_dict[row['video']] = []
        annot_dict[row['video']].append(float(row['iou-stoch']))

# loop through 100 times
mean_iou_list = []
for i in range(100):
    iou_list = []
    # one random draw from every video
    for vid_name in annot_dict.keys():
        iou_list.append(random.choice(annot_dict[vid_name]))
    # then calc the mean IoU for this run.
    mean_iou_list.append(np.array(iou_list).mean())
print(f"Mean: {1-np.array(mean_iou_list).mean()}, "\
      f"stderr: {(1-np.array(mean_iou_list)).std()/np.sqrt(len(mean_iou_list))}")

# find best at every video.
best_ious = []
for vid_name in annot_dict.keys():
    best_ious.append(max(np.array(annot_dict[vid_name])))

print(f"Best: {1-np.array(best_ious).mean()}")
