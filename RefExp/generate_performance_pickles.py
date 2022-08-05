"""Calculates the error at all DRs.

Accepts as command line arguments a scoring method, replacement method, and
split, then pulls models, distributions, object sources from database. Error
and rmae at every coverage is calculated, then this is saved to a pickle with
the arguments as its name.

Typical usage example:
    python generate_performance_pickles.py entropy ensemble_consensus_3 testA
"""
import random
import sys
from collections import Counter
from itertools import groupby
import pickle
import numpy as np
from util.database_commands import \
        convert_boxes_to_corners, get_random_draw,\
        get_re_by_target
from util.database_object import DatabaseObject
from util import calculation_utils
# Pull command line arguments
scoring_method = sys.argv[1]
replacement_method = sys.argv[2]
split = sys.argv[3]

# How many trials to run for each scenario (model, obj source, dist)
num_trials = 100

dbo = DatabaseObject(memory=True)

# Now get all the kinds of models.
query = "SELECT DISTINCT architecture, object_source FROM models"
dbo.cur.execute(query)
architectures_and_sources = dbo.cur.fetchall()

# Group models so we can randomly select results per-run
groups = {}
for result in architectures_and_sources:
    key = f"{result['architecture']}-{result['object_source']}"
    groups[key] = dbo.get_model_ids(result['architecture'], result['object_source'])

saved_runs = {}

# Iterate through every model
for group in groups:
    if len(groups[group]) == 0:
        continue
    for distribution_id in dbo.idx_to_distribution_dict:
        # This tracks errors across all runs and all coverages.
        # Since we don't know its size yet, set to None here.
        if dbo.idx_to_distribution_dict[distribution_id] not in ["softmax", "dropout"]:
            continue
        all_run_errors = None
        print(split, group, distribution_id, scoring_method, replacement_method)

        # Let's get started on our trials.
        for run in range(num_trials):
            # select one output per target, from a random model.
            draw = get_random_draw(
                random.choice(groups[group]), distribution_id, split, dbo.cur)
            # And if there is no response, just skip it.
            if len(draw['target_ids']) == 0:
                break

            # If we haven't created the result arrays yet, do that.
            if all_run_errors is None:
                all_run_errors = np.zeros((num_trials, len(draw['target_ids'])+1))
                all_run_rmaes = np.zeros((num_trials, len(draw['target_ids'])+1))

            all_rq_counts = np.zeros((len(draw['target_ids'])+1), dtype=int)

            # Loop through until DDC.
            # We use a while loop here, since some methods require multiple
            # queries per-update.
            while all_rq_counts.sum() < len(draw['target_ids']):
                # These operations are pretty clear from the method names.
                all_run_errors[run, all_rq_counts.sum()] = calculation_utils.calc_error(draw)
                all_run_rmaes[run, all_rq_counts.sum()] = calculation_utils.calc_rmae(draw)
                to_requery, draw = calculation_utils.get_which_to_requery(
                    draw, scoring_method)
                replacement = get_re_by_target(
                    draw['target_ids'][to_requery], dbo.cur)

                # Update the belief based on the selection fn.

                # In naive replacement, we always use the replacement seed.
                if "naive" in replacement_method:
                    max_depth = int(replacement_method.split("_")[-1])
                    # We shift the RQ count by 1.
                    all_rq_counts[to_requery] += 1
                    # Naive replacement is simple: we can just update the
                    # failure_mode column.
                    draw['failure_modes'][to_requery] = replacement['outputs_failure_mode']
                    # If we've reached max depth, we set the score to a very
                    # low value, so it doesn't get re-queried.
                    if all_rq_counts[to_requery] >= max_depth:
                        draw['score'][to_requery] = -1e6
                    else:
                        draw['score'][to_requery] =\
                                calculation_utils.calc_rejection_score(
                                    replacement['probabilities'],
                                    scoring_method)

                elif "combined" in replacement_method:
                    max_depth = int(replacement_method.split("_")[-1])
                    # in combined replacement, we need to update everything.
                    all_rq_counts[to_requery] += 1

                    # Update the probabilities.
                    draw['probabilities'][to_requery] = \
                            draw['probabilities'][to_requery] *\
                            replacement['probabilities']

                    # Normalize
                    draw['probabilities'][to_requery] = draw['probabilities']\
                            [to_requery]/draw['probabilities'][to_requery].sum()

                    # low value, so it doesn't get re-queried.
                    if all_rq_counts[to_requery] >= max_depth:
                        draw['score'][to_requery] = -1e6
                    else:
                        # Update the rejection score for the next query.
                        draw['score'][to_requery] =\
                                calculation_utils.calc_rejection_score(
                                    draw['probabilities'][to_requery], scoring_method)

                    # Figure out whether or not it's correct.
                    # Start by getting the detected bboxes.
                    converted_boxes = convert_boxes_to_corners(
                        replacement['detections'])
                    # Then calculate the IoUs with the ground truth.
                    ious = calculation_utils.compute_all_IoUs(
                        (replacement['tlx'], replacement['tly'],
                         replacement['brx'], replacement['bry']),
                        converted_boxes)

                    # If correct, set as correct. If not, either leave as
                    # missed detection, or set to undefined.
                    if ious[draw['probabilities'][to_requery].argmax()] >= 0.5:
                        draw['failure_modes'][to_requery] = dbo.failure_to_idx(
                            'correct')
                    elif draw['failure_modes'][to_requery] != dbo.failure_to_idx(
                        'missed_detection'):
                        draw['failure_modes'][to_requery] = dbo.failure_to_idx(
                            'undefined')
                elif "smart" in replacement_method:
                    max_depth = int(replacement_method.split("_")[-1])
                    # Smart replacement is pretty straightforward.
                    # Updates one at a time.
                    all_rq_counts[to_requery] += 1

                    # Get the previous and new scores.
                    score_previous = draw['score'][to_requery]
                    score_new = calculation_utils.calc_rejection_score(
                        replacement['probabilities'], scoring_method)

                    # Set the failure mode based on which score is lower.
                    draw['failure_modes'][to_requery] =\
                            replacement['outputs_failure_mode'] if\
                            score_new < score_previous else\
                            draw['failure_modes'][to_requery]
                    # And then update the score so that it doesn't get
                    # re-re-queried.
                    if all_rq_counts[to_requery] >= max_depth:
                        draw['score'][to_requery] = -1e6
                    else:
                        draw['score'][to_requery] =\
                                calculation_utils.calc_rejection_score(
                                    replacement['probabilities'],
                                    scoring_method)

                elif "ensemble_mean" in replacement_method:
                    # For ensemble mean, we use the name of the replacement
                    # method to set the ensemble size parameter.
                    # i.e., how many seeds we ensemble.
                    ensemble_size = int(replacement_method.split("_")[-1])

                    # We have to update the re-query count by more than one.
                    # The original refexp counts as part of the ensemble.
                    # So, i.e., for an ensemble of size 3, we would have the
                    # original and two re-queries. meaning RQ count goes
                    # up by 2.
                    all_rq_counts[to_requery] += ensemble_size

                    # Get every ensembled solution
                    solutions = []
                    for i in range(ensemble_size):
                        solutions.append(get_re_by_target(
                            draw['target_ids'][to_requery], dbo.cur))

                    # Find the mean of all the queries.
                    new_dist = draw['probabilities'][to_requery].copy()
                    solution = None # Make sure solution has been defined.
                    for solution in solutions:
                        new_dist += solution['probabilities']
                    new_dist = new_dist/(ensemble_size+1)

                    # Figure out whether or not the detection occurred
                    # First by finding the detected bboxes.
                    converted_boxes = convert_boxes_to_corners(
                        solution['detections'])

                    # Then by calculating the IoUs.
                    ious = calculation_utils.compute_all_IoUs(
                        (solution['tlx'], solution['tly'], solution['brx'],
                         solution['bry']), converted_boxes)

                    # If the IoU is greater than 0.5, set to correct.
                    if ious[new_dist.argmax()] >= 0.5:
                        draw['failure_modes'][to_requery] =\
                                dbo.failure_to_idx('correct')
                    elif draw['failure_modes'][to_requery] !=\
                            dbo.failure_to_idx('missed_detection'):
                        # if it's not, either leave it as "missed detection", or
                        # use unknown. This is relevant for RMAE.
                        draw['failure_modes'][to_requery] = dbo.failure_to_idx(
                            'undefined')
                    # And set score to a very low value so it isn't requeried.
                    draw['score'][to_requery] = -1e6
                elif "ensemble_consensus" in replacement_method:
                    # Consensus follows much the same method as mean, but,
                    # obviously, we select via mode instead of argmax mean.

                    # So ensemble size is from the method string, and requery
                    # count updates appropriately.
                    ensemble_size = int(replacement_method.split("_")[-1])
                    all_rq_counts[to_requery] += ensemble_size

                    # Pull n solutions
                    solutions = []
                    for i in range(ensemble_size):
                        solutions.append(get_re_by_target(draw['target_ids'][to_requery], dbo.cur))
                    # Choices are the indices of selected values.
                    choices = [draw['probabilities'][to_requery].argmax()]
                    solution = None # Make sure solution has been defined.
                    for solution in solutions:
                        choices.append(solution['probabilities'].argmax())

                    # Find the IoUs of detected bounding boxes.
                    converted_boxes = convert_boxes_to_corners(solution['detections'])
                    ious = calculation_utils.compute_all_IoUs(
                        (solution['tlx'], solution['tly'], solution['brx'],
                         solution['bry']), converted_boxes)

                    # Select the mode and figure out if it's correct.
                    freqs = groupby(Counter(choices).most_common(), lambda x:x[1])
                    mode_list = [val for val, count in next(freqs)[1]]
                    if ious[random.choice(mode_list)] >= 0.5:
                        draw['failure_modes'][to_requery] = dbo.failure_to_idx(
                            'correct')
                    elif draw['failure_modes'][to_requery] !=\
                            dbo.failure_to_idx('missed_detection'):
                        draw['failure_modes'][to_requery] = dbo.failure_to_idx(
                            'undefined')
                    draw['score'][to_requery] = -1e6

            # Since the ensemble methods don't always divide evenly,
            # add a check to make sure our coverage isn't less than zero.
            if all_rq_counts.sum() < all_run_errors.shape[1]\
               or "ensemble" not in replacement_method:
                all_run_errors[run, all_rq_counts.sum()] =\
                        calculation_utils.calc_error(draw)
                all_run_rmaes[run, all_rq_counts.sum()] =\
                        calculation_utils.calc_rmae(draw)
        # save this run into a big dict
        if all_run_errors is not None:
            saved_runs[f'{split}-{group}-{distribution_id}-{scoring_method}'\
                       f'-{replacement_method}'] = {}
            saved_runs[f'{split}-{group}-{distribution_id}-{scoring_method}'\
                       f'-{replacement_method}']['errors'] = all_run_errors

# And save that big dict into a pickle
with open(f'armae_arrays/{scoring_method}-{replacement_method}-{split}.pickle',
          'wb') as out_pickle:
    pickle.dump(saved_runs, out_pickle)
