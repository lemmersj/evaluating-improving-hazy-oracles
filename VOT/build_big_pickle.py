"""Performs a directory crawl and creates a big compilation pickle for all the
tracked videos. This should speed things up by limiting file access to a single
long access, instead of doing a bunch of listdirs and opening at every inference.

We retain the same structure as the file system, but use dict keys instead of
folders.
"""
import pickle
import os
import sys

# to_save_dict is what is eventually dumped to the pickle file. It is a dict
# of structure to_save_dict[video][annotator][dist_file].

to_save_dict = {}
in_dir = sys.argv[1]

# Start by getting the video names.
video_names = os.listdir(f"{in_dir}")
vid_count = 0

# Loop through every video
for video in video_names:
    annot_count = 0
    vid_count += 1
    print(f"{video} ({vid_count}/{len(video_names)}")
    to_save_dict[video] = {}
    try:
        annotators = os.listdir(f"{in_dir}/{video}")
    except NotADirectoryError:
        continue
    # Loop through every annotator for this video.
    for annotator in annotators:
        annot_count += 1
        print(f"annot {annot_count} of {len(annotators)}")
        to_save_dict[video][annotator] = {}
        dist_files = os.listdir(f"{in_dir}/{video}/{annotator}/")
        for dist_file in dist_files:
            # Skip any non-pickle files
            if "pkl" not in dist_file.split(".")[-1]:
                continue
            # save the data to dump later.
            with open(
                f"{in_dir}/{video}/{annotator}/{dist_file}", "rb") as infile:
                data_in = pickle.load(infile)
                to_save_dict[video][annotator][dist_file] = {}
                to_save_dict[video][annotator][dist_file] = data_in
                to_save_dict[video][annotator][dist_file] = data_in

# dump the data
with open(f"{in_dir}/compiled.pickle", "wb") as out_pickle:
    pickle.dump(to_save_dict, out_pickle)
