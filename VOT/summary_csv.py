"""Produces a summary csv that is accessed by some scripts"""
import os
import pickle
import sys
import numpy as np
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.evaluation import get_dataset
import util

if __name__ == "__main__":
    # start dir is the directory output by build_distributions.py
    start_dir = sys.argv[1]
    print("Getting dataset")
    dataset = get_dataset("otb")

    # Load in all the fit distributions
    print("loading pickle")
    os.makedirs(f"{start_dir}/part_vs_det", exist_ok=True)
    # Load the compiled run data
    with open(f"{start_dir}/compiled.pickle", "rb") as in_pickle:
        compiled_pickle = pickle.load(in_pickle)

    # Each element in each of these lists corresponds to a video.
    # In other words, score_list[i] is the score of the video in vid_list[i]
    num_particles = 1000

    # Track the number of re-queries, so we don't exceed the depth constraint.
    rq_counts = np.zeros(len(dataset), dtype=int)

    vid_count = 0
    all_mean_ious_stoch = []
    all_mean_ious_det = []
    annot_list = []
    vid_list = []

    # Produce a summary csv
    summary_csv = open(f"{start_dir}/part_vs_det/summary.csv", "w")
    summary_csv.write("video,annotator,iou-stoch,iou-det,pct-stoch,pct-det\n")
    # Loop through every video
    for video in dataset:
        vid_count += 1

        # Now we select an annotator.
        # This has already been filtered to only have correct initializations.
        all_annots = list(compiled_pickle[video.name].keys())
        for annot in all_annots:
            vid_list.append(video.name)
            annot_list.append(annot)
            det_iou_list = []
            det_boxes_list = []
            stoch_iou_list = []
            stoch_boxes_list = []
            data = compiled_pickle[video.name][annot]
            particles = \
                util.generate_first_particles(video.name,
                                              annot,
                                              compiled_pickle,
                                              num_particles)

            # For every frame
            for cur_frame_idx in range(len(particles['locs'])):

                stoch_boxes =\
                    particles['locs'][cur_frame_idx][particles['probs'][cur_frame_idx].argmax(), :]
                # Get the IoU
                stoch_iou_list.append(util.calc_frame_iou(np.array(
                    video.ground_truth_rect[cur_frame_idx]),np.array(
                        stoch_boxes)))
            # Calculate the mean IoU and the "detect rate"
            summary_iou = util.calc_vid_iou(video, particles)
            summary_pct_iou_stoch = (np.array(stoch_iou_list) > 0.5).mean()
            all_mean_ious_stoch.append(summary_iou)
            if np.array(stoch_iou_list).mean() != summary_iou:
                print(
                    f"MISMATCH BETWEEN FRAMEWISE AND SUMMARY:"\
                    f"{stoch_boxes_list}, {video.name}, {annot}")
            all_mean_ious_det.append(-1)
            summary_pct_iou_det = -1
            # write the overall summary (video stats)
            summary_csv.write(f"{video.name},{annot},{summary_iou},"\
                              f"{all_mean_ious_det[-1]},{summary_pct_iou_stoch},"\
                              f"{summary_pct_iou_det}\n")

            os.makedirs(f"{start_dir}/part_vs_det/{video.name}/{annot}",
                        exist_ok=True)
            # write the summary for this user (frame stats)
            out_csv = open(f"{start_dir}/part_vs_det/{video.name}/{annot}/summary.csv", "w")
            out_csv.write("frame,stoch,det\n")
            for cur_frame in range(len(video.ground_truth_rect)):
                out_csv.write(
                    f"{cur_frame},{stoch_iou_list[cur_frame]},"\
                    f"{det_iou_list[cur_frame]}\n")
            out_csv.close()
    summary_csv.close()
