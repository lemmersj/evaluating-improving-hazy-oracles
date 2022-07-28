"""Calculate the best accuracy of all models.

Table 1, as of the time this comment was written

Typical usage:
    python calc_acc_best_and_worst.py
"""
import sqlite3
import numpy as np
from util.database_commands import adapt_array, convert_array,\
        convert_reshape_array,\
        get_failure_mode_dict, get_failure_mode_dict_reversed,\
        get_distribution_dict, get_outputs_one_run,\
        get_target_from_sentence

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

# Now run the MAE/AMAE calculation.
# for AMAE we assume that the failure mode has missed detections annotated.
# Loop through every model type
for group in groups:
    # and through every distribution
    for distribution_id in distribution_ids:
        # and split
        for split in ["val", "testA", "testB"]:
            if distribution_dict[distribution_id] != "dropout":
                continue
            # finding best and worst for each model.
            this_model_best = []
            this_model_worst = []
            for model in groups[group]:
                run = get_outputs_one_run(model, split, distribution_id, cur)
                correct_exists_dict = {}
                incorrect_exists_dict = {}
                for row in run:
                    target = get_target_from_sentence(row['sentence'], cur)
                    if target not in correct_exists_dict:
                        correct_exists_dict[target] = False
                        incorrect_exists_dict[target] = False
                    if failure_dict_reversed[row['failure_mode']] == 'correct':
                        correct_exists_dict[target] = True
                    else:
                        incorrect_exists_dict[target] = True
                correct_exists_list = [int(correct_exists_dict[key]) for key in correct_exists_dict]
                incorrect_exists_list = [int(incorrect_exists_dict[key]) for\
                                         key in incorrect_exists_dict]
                this_model_best.append(np.array(correct_exists_list).mean())
                this_model_worst.append(1-np.array(incorrect_exists_list).mean())
            print(f"{group}-{distribution_dict[distribution_id]}-{split}"\
                  f"-best:{np.array(this_model_best).mean()}+-"\
                  f"{np.array(this_model_best).std()/np.sqrt(len(this_model_best))}")
            #print(f"{group}-{distribution_dict[distribution_id]}-{split}"\
            #      f"-worst:{np.array(this_model_worst).mean()}+-"\
            #      f"{np.array(this_model_worst).std()/np.sqrt(len(this_model_worst))}")
