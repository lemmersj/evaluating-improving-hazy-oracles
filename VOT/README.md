# Deferred Inference (VOT)
This is a repository for evaluating deferred inference on the single-target VOT task. All of the files in this directory should be dropped into the pytracking directory of the [pytracking](https://github.com/visionml/pytracking) repository. Conda environment is stored in environment.yaml

The instructions here begin with a 100 csv files (corresponding to 100 stochastic passes) for every video/annotator pair. These should be stored as:

tracker\_outputs/{vid\_name}/{mturk/gold\_standard}/stoch\_True)/{annotator}-{run\_num}.csv. Columns are frame, gt\_tlx, gt\_tly, gt\_w, gt\_h, guess\_tlx, guess\_tly, guess\_w, guess\_h.

## Aggregating stochastic passes
We would like to group stochastic passes together as MoG distributions.

    python build_distributions.py 10 20 True

`build_distributions.py` is additionally used with `pick_dbscan_params.py` to perform the gridsearch for DBSCAN parameters. To do this, first follow the instructions for calculating the random and err @ 0 errors.

## Compiling aggregated passes.
We do a second level of aggregation to put all of the stochastic MoGs into a single pickle, making it much faster to load. this is done via:

		python build_big_pickle.py [output_dir]

## Performing Runs
We next perform runs (100 random draws from the roughly 9<sup>100</sup> initialization combinations) using this aggregated data. This is done with:

    python perform_runs.py selection_fn run_number

This produces a numpy array of dimension 10 x 101 x 100 for every DR and DD, where only the appropriate elements are filled in. That is, all the saved numpy arrays can be added together to produce a full array. This was done to allow simple parallelism.

## Calc deferral-free and and perfect deferral errors (Table 1)
Create a summary csv file:

    python summary_csv.py output_videos_wh_correlated/10.0-20-True

Analyze it for deferral-free and perfect deferral.

    python no_deferral.py

## Calc DEV, err @ 1 and marginals (Table 2 and Figure 5)
    python calc_dev.py

## Produce all error-DR plots (supplemental)
    python plot_all.py
