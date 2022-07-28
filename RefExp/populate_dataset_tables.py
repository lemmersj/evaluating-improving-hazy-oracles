"""
Produces and populates dataset tables for RefExp seed rejection eval.

This should be with no data/redatabase.sqlite3 file, then this file should
be moved into the folder for the appropriate REC algorithm. The schema is
available in the README.md file.

No command line arguments.

typical usage:
    python populate_dataset_tables.py
"""

import sqlite3
import pickle
import os
import json
from util import database_commands
import sys

# Change as necessary.
mscoco_train_location = "/path/to/mscoco/annotations/instances_train2014.json"
mscoco_val_location = "/path/to/mscoco/annotations/instances_val2014.json"

# Load in the MSCOCO json
with open(mscoco_train_location, "r") as in_file:
    mscoco_train_json = json.load(in_file)

with open(mscoco_val_location, "r") as in_file:
    mscoco_val_json = json.load(in_file)

# Create a dict where every annotation id keys to the corresponding bbox.
mscoco_dict = {}
for row in mscoco_train_json['annotations']:
    if row['id'] in mscoco_dict.keys():
        print("I guess ids are redundant...")
        sys.exit()
    mscoco_dict[row['id']] = row['bbox']

for row in mscoco_val_json['annotations']:
    if row['id'] in mscoco_dict.keys():
        print("I guess ids are redundant...")
        sys.exit()
    mscoco_dict[row['id']] = row['bbox']

# Create the database file
command = "sqlite3 data/redatabase.sqlite3 < data/ReDatabase.sql"

# I've read some warnings that I should use subprocess instead of os,
# but that call seems to struggle with the syntax of this command.
os.system(command)

with open("data/refs(unc).p", "rb") as in_data:
    data = pickle.load(in_data)

# add the distributions
con = sqlite3.connect("data/redatabase.sqlite3")
cur = con.cursor()
cur.execute("INSERT INTO distributions(name) VALUES (?)", ("softmax",))
cur.execute("INSERT INTO distributions(name) VALUES (?)", ("varratio",))
cur.execute("INSERT INTO distributions(name) VALUES (?)", ("dropout",))
cur.execute("INSERT INTO distributions(name) VALUES (?)", ("dropout_textonly",))
cur.execute("INSERT INTO distributions(name) VALUES (?)", ("varratio_textonly",))
con.commit()

# add the failure modes
cur.execute("INSERT INTO failure_modes(name) VALUES (?)", ("undefined",))
cur.execute("INSERT INTO failure_modes(name) VALUES (?)", ("missed_detection",))
cur.execute("INSERT INTO failure_modes(name) VALUES (?)", ("ambiguous",))
cur.execute("INSERT INTO failure_modes(name) VALUES (?)", ("misunderstood",))
cur.execute("INSERT INTO failure_modes(name) VALUES (?)", ("correct",))
con.commit()

# Add the rows.
i = 0
for row in data:
    #print(i, len(data))
    i+=1
    database_commands.add_rows_from_dataset_entry(row, mscoco_dict, cur)
con.commit()
con.close()
