"""Calculates the error at all constraints.

Data is saved as a pickle containing a numpy array of dimension 10 x 101 x 100.
The first dimension is the DDC, the second is the error @ rqr, and
the third is the run number. The output pickle only fills in one element of
the last dimension, and all output files can simply be summed together,
allowing for parallelization.

typical usage:
    python calc_armae.py <method> <run_number>

"""
import os
import pickle
import sys
import random
from PIL import Image
import util
import copy
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
import numpy as np
from pytracking.evaluation import get_dataset

def do_pass(compiled_pickle, depth_constraint, cur_pass, method, dataset):
    """a single re-query pass.

    args:
        compiled_pickle: the compiled pickle data
        depth_constraint: the depth constraint
        cur_pass: the current pass
        method: the re-query methods
        dataset: the formatted dataset.

    returns:
        a dict containing depth_constraint, num_pass, and error at every RQR
    """
    # Each element in each of these lists corresponds to a video.
    # In other words, score_list[i] is the score of the video in vid_list[i]
    score_list = []
    vid_list = []
    iou_list = []
    rqr_ious = []
    frame_size_list = []
    dist_list = [] # for ensemble mean
    particles_list = []
    num_particles = 10000
    num_score_samples = 500

    # Track the number of re-queries, so we don't exceed the depth constraint.
    rq_counts = np.zeros(len(dataset), dtype=int)

    # Do the initial run. Start by looping over all videos.
    for video in dataset:
        # Now we select an annotator for this video.
        all_annots = list(compiled_pickle[video.name].keys())
        annot_1 = random.choice(all_annots)

        # Data is the distribution for the video and annotator.
        data = compiled_pickle[video.name][annot_1]
        dist_list.append(data)

        # Keep track of video frame sizes.
        with Image.open(video.frames[0]) as in_image:
            frame_size_list.append(in_image.size)

        print(f"Drawing initial particles {depth_constraint}-{cur_pass}")
        # Generate and smear samples
        if "combined" not in method:
            particles_list.append(
                util.generate_first_particles(video.name,
                                              annot_1,
                                              compiled_pickle,
                                              num_particles))
        else:
            base_particles = util.generate_first_particles(video.name,
                                                           annot_1,
                                                           compiled_pickle,
                                                           num_particles//2)
            smeared_particles = util.smear_particles(
                copy.deepcopy(base_particles), data, (7,7,7,7))
            for cur_frame in range(len(base_particles['locs'])):
                base_particles['locs'][cur_frame] = np.concatenate((
                    base_particles['locs'][cur_frame],
                    smeared_particles['locs'][cur_frame]))
                base_particles['probs'][cur_frame] = np.concatenate((
                    base_particles['probs'][cur_frame],
                    smeared_particles['probs'][cur_frame]))
            particles_list.append(base_particles)

        # save the score
        score_list.append(util.calc_score(
            video, particles_list[-1], "mean", num_score_samples))

        # and ious
        iou_list.append(util.calc_vid_iou(video, particles_list[-1]))

        # Keep track of videos in the pytracking library form.
        vid_list.append(video)

    # And calculate the IoU at zero DR.
    rqr_ious.append(np.array(iou_list).mean())
    # We've drawn the first set of human inputs. Now start doing the replacement.
    for i in range(len(iou_list)):
        # select the video to re-query. The one with the lowest score.
        vid_to_replace = np.array(score_list).argmin()

        print(f"Iteration {i} replacing {vid_to_replace}")

        # Select another annotator.
        # Replacement is allowed
        all_annots = list(compiled_pickle[vid_list[vid_to_replace].name].keys())
        annot_2 = random.choice(all_annots)
        data = compiled_pickle[vid_list[vid_to_replace].name][annot_2]

        if "naive" in method:
            # For naive replacement, just update score and iou to the re-queried
            # value
            rq_counts[vid_to_replace] += 1
            samples = util.generate_first_particles(vid_list[vid_to_replace].name,
                                               annot_2, compiled_pickle,
                                               num_particles)
            new_iou = util.calc_vid_iou(vid_list[vid_to_replace], samples)
            new_score = util.calc_score(
                vid_list[vid_to_replace], samples, "mean", num_score_samples)
            score_list[vid_to_replace] = new_score
            iou_list[vid_to_replace] = new_iou
        elif "smart" in method:
            # for smart replacement...
            rq_counts[vid_to_replace] += 1

            # Create the new samples and the score.
            samples = util.generate_first_particles(vid_list[vid_to_replace].name,
                                               annot_2, compiled_pickle,
                                               num_particles)
            new_score = util.calc_score(
                vid_list[vid_to_replace], samples, "mean", num_score_samples)
            # And if the score is better (higher) than the previous score,
            # perform the replacement.
            if new_score > score_list[vid_to_replace]:
                score_list[vid_to_replace] = new_score
                iou_list[vid_to_replace] = util.calc_vid_iou(
                    vid_list[vid_to_replace], samples)
        elif "combined" in method:
            # For combined replacement...
            rq_counts[vid_to_replace] += 1
            vid_name = vid_list[vid_to_replace].name
            # Pull the new dist
            dist = compiled_pickle[vid_name][annot_2]
            # And update the samples
            particles_list[vid_to_replace] = util.update_probs(dist,
                                                              particles_list[
                                                                  vid_to_replace])
            # Calculate the score and iou given the new samples.
            # and update the main lists.
            score_list[vid_to_replace] = util.calc_score(
                vid_list[vid_to_replace],
                particles_list[vid_to_replace], "mean", num_score_samples)
            new_vid_iou = util.calc_vid_iou(
                vid_list[vid_to_replace], particles_list[vid_to_replace])
            iou_list[vid_to_replace] = new_vid_iou
        elif "ensemble_mean" in method:
            # We can't just rely on the outer loop, because every iteration
            # adds multiple deferrals.
            # So instead, count the number of deferrals, and break when
            # DR=1
            if np.array(rq_counts).sum() > len(iou_list):
                rqr_ious = rqr_ious[:len(iou_list) + 1]
                break

            # Increment RQ counts.
            rq_counts[vid_to_replace] += depth_constraint

            # The process for deferral is to pull N distributions,
            # each of which is stored in the these_dists list.
            these_dists = []
            # Add the original human input.
            these_dists.append(dist_list[vid_to_replace])

            # and add the number of required deferrals
            for _ in range(depth_constraint):
                annot_2 = random.choice(all_annots)
                these_dists.append(
                    compiled_pickle[vid_list[vid_to_replace].name][annot_2])

            # Sample from every one of the distributions in these_dists
            cur_samples = None
            for dist in these_dists:
                new_samples = util.particles_from_dist(dist, num_particles)
                if cur_samples is None:
                    cur_samples = new_samples
                else:
                    for cur_frame in range(len(new_samples['locs'])):
                        cur_samples['locs'][cur_frame] = np.concatenate(
                            (new_samples['locs'][cur_frame],
                             cur_samples['locs'][cur_frame]))
                        cur_samples['probs'][cur_frame] = np.concatenate(
                            (new_samples['probs'][cur_frame],
                             cur_samples['probs'][cur_frame]))
            sorted_dist_indices = sorted(these_dists[0].keys())

            # Now that we have all the samples, find the mean prob under
            # all distributions
            for cur_frame in range(len(cur_samples['locs'])):
                cur_samples['probs'][cur_frame] = np.zeros(
                    cur_samples['probs'][cur_frame].shape)
                for dist in these_dists:
                    scores_this_dist = dist[
                        sorted_dist_indices[cur_frame]].score_samples(
                            cur_samples['locs'][cur_frame])
                    scores_this_dist = np.exp(scores_this_dist)
                    #scores_this_dist = scores_this_dist/scores_this_dist.sum()
                    cur_samples['probs'][cur_frame] += scores_this_dist/len(these_dists)
            iou_list[vid_to_replace] = util.calc_vid_iou(vid_list[vid_to_replace], cur_samples)
        else:
            print("Invalid method specified")
            sys.exit()

        # Manually implement depth constraint
        if rq_counts[vid_to_replace] >= depth_constraint:
            score_list[vid_to_replace] = 1e6

        # Update the metric. For ensemble mean there's more than one step.
        if "ensemble_mean" in method:
            rqr_ious += [np.array(iou_list).mean()]*depth_constraint
        else:
            rqr_ious.append(np.array(iou_list).mean())

    return {'depth_constraint': depth_constraint,
               'cur_pass': cur_pass,
               'rqr_ious':rqr_ious}

if __name__ == "__main__":
    method = sys.argv[1]
    cur_pass = int(sys.argv[2])
    dataset = get_dataset("otb")

    # Load in all the fit distributions
    with open(
        "output_videos_wh_correlated/10.0-20-True/compiled.pickle", "rb") \
            as in_pickle:
        compiled_pickle = pickle.load(in_pickle)

    # Depth constraint, RQR, pass.
    saved_passes = np.zeros((10, 101, 100))

    # Loop through depth constraints
    for depth_constraint in range(1, 11):
        print(f"run {cur_pass} constraint {depth_constraint}")
        cur_sample = do_pass(compiled_pickle, depth_constraint,
                             cur_pass, method, dataset)
        saved_passes[cur_sample['depth_constraint']-1,
                     :len(cur_sample['rqr_ious']),
                     cur_sample['cur_pass']] = 1-np.array(cur_sample['rqr_ious'])
    with open(f"better_smear/{method}-{cur_pass}.pkl", "wb") as outfile:
        pickle.dump(saved_passes, outfile)
