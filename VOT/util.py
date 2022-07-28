"""utility functions for deferral analysis on VOT."""
import numpy as np
from numba import njit

def calc_score(video, particles, method, num=100):
    """Calculates the rejection score for a video.

    args:
        video: the video in pytracking form
        particles: the particles
        method: the method for calculating scores.
        num: the number of samples for which IoU is calculated.

    returns:
        the rejection score
    """
    score_list = []
    # Iterate through all frames
    for frame_num in range(len(particles)):
        if particles['probs'][frame_num].sum() != 1:
            # Calculate normalized probabilities.
            probabilities = particles['probs'][frame_num]-particles['probs'][frame_num].max()
            probabilities = np.exp(probabilities)
            probabilities = probabilities/probabilities.sum()
        # Pick n bounding boxes using weighted random sampling
        selection = np.random.choice(
            [*range(len(particles['probs'][frame_num]))], size=num, p=probabilities)
        # Calculate the ious
        location = particles['locs'][frame_num][selection, :]
        given_sample_ious = sample_ious(location, num)

        # and calculate the scores
        if method == "mean":
            score_list.append(np.mean(np.array(given_sample_ious)))

    return np.mean(np.array(score_list))

def update_probs(new_dist, particles):
    """Update the probabilities for particles under a new distribution.

    args:
        new_dist: the new distribution
        particles: the particles

    returns:
        the particles with their scores updated.
    """
    sorted_keys = sorted(new_dist.keys())
    for cur_frame in range(len(particles['locs'])):
        new_probs = new_dist[sorted_keys[cur_frame]].score_samples(
            particles['locs'][cur_frame])
        particles['probs'][cur_frame] += new_probs
    # It probably passes by reference, but return anyway.
    return particles

def calc_vid_iou(video, particles):
    """Calculates the iou for a whole video.

    args:
        video: the video in pytracking format
        particles: the particles.

    returns:
        the mean per-frame IoU"""

    frame_ious = []
    for frame_idx in range(len(particles['locs'])):
        gt_bbox = video.ground_truth_rect[frame_idx]
        candidate_bbox = particles['locs'][frame_idx][particles['probs'][frame_idx].argmax()]
        frame_ious.append(calc_frame_iou(np.array(gt_bbox), np.array(candidate_bbox)))
    return np.array(frame_ious).mean()

def create_uniform_particles(max_x, max_y, count):
    """Creates uniform particles within a range.

    args:
        max_x: the maximum x value
        max_y: the maximum y values
        count: how many particles to create.

    returns:
        tuples of tlx tly brx bry
    """
    # TODO: I think this was left behind from previous code. Can I delete this function?

    # produce random vals
    random_vals = np.random.random((count, 4))

    # scale it to xy
    random_vals[:, 0] = random_vals[:, 0]*max_x
    random_vals[:, 1] = random_vals[:, 1]*max_y

    # and produce a brx bry
    random_vals[:, 2] = (max_x-random_vals[:, 0])*random_vals[:,2]
    random_vals[:, 3] = (max_y-random_vals[:, 1])*random_vals[:,3]

    return random_vals

def smear_particles(particles, dist, stdevs):
    """Adds gaussian noise to particles

    args:
        particles: the particles to add noise to.
        dist: the distributions, used for finding the updated probabilities.
        stdevs: the standard deviations by which to smear.

    returns:
        particles with gaussian noise added.
    """
    # Do some reshaping so we can sample
    stdevs_reshaped = np.array(stdevs).reshape(1, -1).repeat(
        particles['locs'][0].shape[0], axis=0)

    # for every frame...
    sorted_keys = sorted(dist.keys())
    for frame in range(len(particles['locs'])):
        # add random noise to location
        rand_draw = np.random.randn(particles['locs'][frame].shape[0], 4)
        particles['locs'][frame] += rand_draw*stdevs_reshaped

        # and update probability.
        particles['probs'][frame] = dist[sorted_keys[frame]].score_samples(
            particles['locs'][frame])

    return particles

def generate_first_particles(video_name, annotator, compiled_pickle, num_particles):
    """Randomly generates the first set of particles for a video.

    args:
        video_name: the name of the video (string)
        annotator: the annotator id (string)
        compiled_pickle: the pickle data
        num_particles: how many particles to use.

    returns:
        a list of particles (a dict with keys 'locs' and 'probs',
        each of which is a list of length num_frames)"""
    cur_vid = compiled_pickle[video_name][annotator]
    sorted_keys = sorted(cur_vid.keys())
    frame_locs = []
    frame_probs = []

    # Loop through every frame
    for key in sorted_keys:
        # and sample locations
        locs = cur_vid[key].sample(num_particles)[0]
        # Then calculate the probabilities.
        probs = cur_vid[key].score_samples(locs)

        frame_locs.append(locs)
        frame_probs.append(probs)

    return {'locs': frame_locs, 'probs': frame_probs}

def particles_from_dist(dist, num_particles):
    """randomly samples a set of particles from every distribution.

    args:
        dist: the distribution
        num_particles: the number of particles

    returns:
        a set of particles (a dict with keys 'locs' and 'probs',
        each of which is a list of length num_frames)
    """

    sorted_keys = sorted(dist.keys())

    frame_locs = []
    frame_probs = []
    # for every frame
    for key in sorted_keys:
        # sample from the distributions to produce a location
        locs = dist[key].sample(num_particles)[0]
        # find and normalize the probabilities
        probs = np.exp(dist[key].score_samples(locs))
        probs = probs/probs.sum()

        frame_locs.append(locs)
        frame_probs.append(probs)

    return {'locs': frame_locs, 'probs': frame_probs}

@njit
def sample_ious(bboxes, num=100):
    """Calculate IoU between a number of random bounding boxes.

    args:
        boxes: a list of bounding boxes in tlx tly w h
        num: the number of samples

    returns:
        a list of num ious between randomly chosen bounding boxes.
    """
    ious = []
    # for every query
    for i in range(num):
        # select a random first and second bbox
        rand_1 = np.random.randint(0, bboxes.shape[0])
        rand_2 = np.random.randint(0, bboxes.shape[0])

        # and calculate the IoU
        box_1 = bboxes[rand_1, :]
        box_2 = bboxes[rand_2, :] # boxes should be tlx tly w h
        ious.append(calc_frame_iou(box_1, box_2))
    return ious

@njit
def calc_frame_iou(ground_truth, target):
    """Calculates the IoU between two bounding boxes.

    args:
        ground_truth: bbox in tlx, tly, w, h
        target: box in tlx, tly, w, h

    returns:
        float IoU between boxes."""

    # convert to tlx tly brx bry
    new_ground_truth = np.zeros(4)
    new_target = np.zeros(4)
    new_ground_truth[2:] = ground_truth[:2] + ground_truth[2:]
    new_ground_truth[:2] = ground_truth[:2]
    new_target[2:] = target[:2] + target[2:]
    new_target[:2] = target[:2]

    ground_truth = new_ground_truth
    target = new_target

    intersection_tlx = max(ground_truth[0], target[0].item())
    intersection_tly = max(ground_truth[1], target[1].item())
    intersection_brx = min(ground_truth[2], target[2].item())
    intersection_bry = min(ground_truth[3], target[3].item())

    # manual case for zero IoU.
    if intersection_tlx > intersection_brx or intersection_tly > intersection_bry:
        return 0

    intersection_area = (intersection_brx - intersection_tlx)*(intersection_bry - intersection_tly)
    union_area = (ground_truth[2]-ground_truth[0])*(
        ground_truth[3]-ground_truth[1])+(target[2]-target[0])*(
            target[3]-target[1])-intersection_area

    return (intersection_area/union_area).item()
