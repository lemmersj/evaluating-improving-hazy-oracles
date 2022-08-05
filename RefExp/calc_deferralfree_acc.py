"""Calculate the deferral-free accuracy of all models.

Table 1, as of the time this comment was written.

Typical usage:
    python calc_acc_pertarget_rand.py
"""
import sqlite3
import random
import numpy as np
from util.database_commands import adapt_array, convert_array,\
        convert_reshape_array, get_failure_mode_dict,\
        get_failure_mode_dict_reversed, get_distribution_dict

# How many trials to run for each scenario (model, obj source, dist)
num_trials = 100

# Connect to the database
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("detections", convert_reshape_array)
sqlite3.register_converter("probabilities", convert_array)
con_tmp = sqlite3.connect("data/redatabase.sqlite3", detect_types=sqlite3.PARSE_COLNAMES)

# Copy to memory, for faster access.
con = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_COLNAMES)
con.row_factory = sqlite3.Row
con_tmp.backup(con)
con_tmp.close()

# This line allows us to access the query results as a dict
con.text_factory = lambda b: b.decode(errors='ignore')

# And create the cursor.
cur = con.cursor()

# Get the failure mode dict for lookup
failure_dict_reversed = get_failure_mode_dict_reversed(cur)
failure_dict_forward = get_failure_mode_dict(cur)
distribution_dict = get_distribution_dict(cur)

# Now get all the kinds of models.
query = "SELECT DISTINCT architecture, object_source FROM models"
cur.execute(query)
architectures_and_sources = cur.fetchall()

# And all of the distributions
query = "SELECT id FROM distributions"
cur.execute(query)
distribution_ids = cur.fetchall()
distribution_ids = [distribution_id[0] for distribution_id in distribution_ids]

# Get grouped models
groups = {}
for result in architectures_and_sources:
    key = f"{result['architecture']}-{result['object_source']}"
    if result['object_source'] is not None:
        query = "SELECT id FROM models WHERE architecture=? AND object_source=?"
    else:
        query = "SELECT id FROM models WHERE architecture=?"
        result = (result[0],)
    cur.execute(query, (result))
    model_ids = cur.fetchall()
    groups[key] = [model_id[0] for model_id in model_ids]

# Query to select a random element for every target
query = "SELECT outputs_id as id, outputs_failure_mode as failure_mode "\
        "FROM (SELECT outputs.id as outputs_id, outputs.failure_mode as "\
        "outputs_failure_mode, sentences.target as sentences_target FROM "\
        "outputs JOIN sentences ON outputs.sentence = sentences.id WHERE "\
        "model=? AND distribution=? AND split=? ORDER BY random()) GROUP BY "\
        "sentences_target"

# We save the results to a csv
with open("accuracies_random.csv", "w") as outfile:
    outfile.write("network,distribution,split,mean,stderr\n")

# Loop through every split
for split in ["val", "testA", "testB"]:
    # and every model.
    for group in groups:
        # If the analysis has failed, don't do some things
        for distribution_id in distribution_ids:
            analysis_failed = False
            mae_results = []
            acc_results = []

            # Run a bunch of trials
            for run in range(num_trials):
                # Pick a model
                model = random.choice(groups[group])
                # Could probably have done calculations with some clever sql,
                # but I'd rather just pull the data once and handle it in here.
                cur.execute(query, (model, distribution_id, split))
                data_subset = cur.fetchall()
                if len(data_subset) == 0:
                    analysis_failed = True
                    break
                is_correct = [int(failure_dict_forward['correct'] ==\
                                  datapoint['failure_mode']) for datapoint\
                              in data_subset]
                acc_results.append(np.array(is_correct).mean())
            if analysis_failed:
                continue
            with open("accuracies_random.csv", "a") as outfile:
                outfile.write(f"{group},{distribution_dict[distribution_id]},"\
                              f"{split},{np.array(acc_results).mean()},"\
                              f"{np.array(acc_results).std()/np.sqrt(len(acc_results))}\n")
