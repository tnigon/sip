Tyler Nigon
2020-11-17

# SLURM array job scheduling

Guidelines for how to submit an array of jobs to MSI so hundreds of SLURM scripts do not have to be made separately.

Keep in mind that images must be processed before tuning/training/testing can be performed on the spectral features.
Thus, it is important that all images are processed before tuning, but also keep in mind that there is limited high
performance storage space and data has to be managed so that HPS doesn't go above our 5 TB limit. Because data are
transferred to 2nd Tier Storage upon completion of tuning/training/testing, the sooner we complete the tuning, the
sooner space will open up for further processing.

## Image processing scenarios

There are several levels of image processing, and we only have to repeat this processing if the data are moved to
2nd Tier Storage before all relevant tuning is completed, so we want to avoid moving to 2nd Tier until all tuning
is complete. It will take a bit of planning to be sure all this is done in the correct order while balancing the
HPS limit and avoiding processing multiple times.

To avoid processing files in the same way at the same time, care should be taken not to set up a job array so that
there is any chance for overlap. I think the easiest way to avoid multiple jobs at the same time is to just keep the
step size at least 54 for "proc_img" (at the "clip" level). Another way would be to do a "setup" proc_img script to
process every 54th scenario at the very beginning, which allows a minimum step size of 27. The "setup" proc_img
script was further modified to run for both "smooth" scenarios, to allow a minimum "tune_train" step size of 9.

## run the proc_img "setup"

`$ sbatch --array=1-648:54 sip_2_procimg_setup.sh`

This command will process all "clip" and "smooth" scenarios ("dir_panels" and "crop" are already complete). Thus,
the only scenarios left to process are "bin" and "segment", and the minimum step size for "tune_train" is 9.


## run proc_img for a group of scenarios

WARNING: Be careful not to run too many scenarios, because HPS space will fill up. It is probably wise to only run
162 scenarios at at time (the "clip" level - this is 25% of all files).

The following will run 18 jobs (9 scenarios per array) for the first 162 scenarios:

`$ sbatch --array=1-162:9 sip_2_procimg_array.sh`

After each of these 18 jobs completed, there was 3.05 TB storage space used on the HPS (before running any
tune_train scenarios). All 18 jobs were complete within 34 minutes (ntasks=24; nodes=1).


## run tune_train for all scenarios with images processed

WARNING: Be sure all relevant images are processed before begining "tune_train". You can check
"msi_XX_time_imgproc.csv" and "msi_2_imgproc_n_files.csv" to be sure all files have been processed (these files are
located in the ../results/msi_XX_results_meta/ directory.

`$ sbatch --array=1-162:3 sip_2_tune_train_array.sh`

If you wish to submit the job(s) right away without waiting for the processing to complete, use the `--depend` or
`--begin` parameter to delay when the job(s) will begin. Be sure to pass the appropriate job/task ID.

`$ sbatch --array=1-162:9 --depend=afterok:123456 sip_2_tune_train_array.sh`

`$ sbatch --array=1-162:9 --begin=now+2hour sip_2_tune_train_array.sh`

NOTE: The `scontrol` command can be used to change the `--begin` time.


## Setup, proc_img, and tune_train

The following three commands can be used right in a row to process 1/4 of all scenarios, as well as tune/train upon
completion of image processing.

```
$ sbatch --array=1-648:54 sip_2_procimg_setup.sh
$ sbatch --array=1-162:9 --depend=afterok:1234 sip_2_procimg_array.sh
$ sbatch --array=1-162:9 --depend=afterok:5678 sip_2_tune_train_array.sh`
```


## Other helpful commands

### Job status and cancelling

To see all the jobs under a particular task array:

`$ squeue --job=123456`


To cancel a single job under a task array:

`$ scancel 123456+1`


To cancel all jobs under a task array:

`$ scancel 123456`


To see the job accounting information and CPU allocations:

`$ sacct -j 123456 --allocations`


### Testing 

To validate the script and return an estimate of when a job would be scheduled to run, pass `--test-only`:

`$ sbatch --array=1-162:9 sip_2_tune_train_array.sh`
