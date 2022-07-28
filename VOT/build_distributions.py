"""
Performs the EM algorithm to group together droout passes, and visualizes.

Every video frame for every annotator has 3 outputs:
    1. Video frame with ground-truth bbox, argmax initialized bbox, and
    all dropout bboxes.
    2. Video frame with ground-truth bbox, argmax initialized bbox, heatmap, and
    max bbox
    3. npz file containing heatmap and gmm parameters.

    These are saved in the structure:
        video_name->annotator_id->boxes_only_{number}.jpg
        video_name->annotator_id->with_heatmap_{number}.jpg
        video_name->annotator_id->dists_{number}.npz
"""
import os
import csv
import pickle
import sys
import util
from sklearn.cluster import DBSCAN
import numpy as np
import sklearn.mixture

def em_unknown_means(vals, eps, min_samples):
    """Performs expectation maximization with unknown # of means.

    args:
        vals: the values to fit
        eps: the epsilon from DBSCAN
        min_samples: minimum samples from DBSCAN

    returns:
        the best fit gmm"
    """
    # perform dbscan
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(vals)
    # and count the clusters
    num_clusters = len(np.unique(db.labels_))
    gmm = sklearn.mixture.GaussianMixture(num_clusters)

    # if they're all outliers, just fit them as is
    if (db.labels_ == -1).all():
        gmm.fit(vals)
    else:
        # Fit only the inliers
        gmm.fit(vals[np.where(db.labels_ != -1)[0], :])

    return gmm

# Get a list of all the videos.
directory = "tracker_outputs"
videos = sorted(os.listdir(directory))

eps = float(sys.argv[1])
min_samples = int(sys.argv[2])
remove_outliers = True #bool(sys.argv[3])

for video in videos:
    # Get the full file path of the video annotations
    full_path_stoch = os.path.join(directory, video, "mturk", "stoch_True")

    # I think we're just using this to get the annotator ids for this specific
    # video, so only need to run listdir once.
    all_csvs = os.listdir(full_path_stoch)
    annotators = []

    # Extract unique annotator ids
    for annotator in all_csvs:
        if annotator.split("-")[0] not in annotators:
            annotators.append(annotator.split("-")[0])

    # for every annotator that did this video
    for annotator in annotators:
        print(f"{video}-{annotator}")
        # All tracks is a nested list of each of the stochastic (dropout)
        # tracks.
        all_tracks = []

        # Check on first annotator instance if the init is good.
        first_frame_first_pass = True
        skip_annotator = False

        # TODO: Make this not a magic number.
        # Looping through every stochastic pass
        for i in range(100):
            # Save this track.
            this_track = []
            gt_track = []

            # Open the track csv.
            filename = os.path.join(full_path_stoch, f"{annotator}-{i}.csv")
            with open(filename, "r") as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    # An initial version saved the csvs with brx and bry
                    # instead of width and height.
                    if "guess_w" not in row.keys():
                        row['gt_w'] = row['gt_brx']
                        row['gt_h'] = row['gt_bry']
                        row['guess_w'] = row['guess_brx']
                        row['guess_h'] = row['guess_bry']

                    this_track.append((float(row['guess_tlx']),
                                      float(row['guess_tly']),
                                      float(row['guess_w']),
                                      float(row['guess_h'])))
                    gt_track.append([float(row['gt_tlx']),
                              float(row['gt_tly']),
                              float(row['gt_w']),
                              float(row['gt_h'])])

                    # Skip any videos where the first-frame IoU < 0.5
                    # since this likely indicates tracking the wrong object.
                    if first_frame_first_pass:
                        first_frame_first_pass = False
                        if util.calc_frame_iou(np.array(this_track[0]),
                                               np.array(gt_track[0])) < 0.5:
                            print(f"Skipping {video}-{annotator}")
                            skip_annotator = True
                            break
            all_tracks.append(this_track)
            if skip_annotator:
                break
        if skip_annotator:
            continue
        all_tracks_stack = np.stack(all_tracks)

        video_dir = video.split("_")[0]
        file_list = sorted(os.listdir(f"/z/dat/OTB2015/{video_dir}/img/"))
        outdir = f"output_videos_wh_correlated/{eps}-{min_samples}-"\
                f"{remove_outliers}/{video}/{annotator}"
        try:
            os.makedirs(outdir)
        except:
            continue

        # do a fit for every frame
        for frame in range(all_tracks_stack.shape[1]):
            # Now we need to EM the bounding boxes.
            dist = em_unknown_means(
                all_tracks_stack[:, frame, :], eps, min_samples)

            # Save peaks and dists for this frame.
            with open(f"{outdir}/dists-{file_list[frame]}.pkl", "wb") as outfile:
                pickle.dump(dist, outfile)
