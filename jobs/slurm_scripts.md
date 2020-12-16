# Run 2 slurm commands

The following commands were passed to Mangi (amdsmall). The job ID of each is recorded.

Run date: 2020-11-20

Running data load: 0.15 TB

## Setup (~20 minutes)
sbatch --array=1-648:54 sip_2_procimg_setup.sh
job 95062
Running data load: 0.78 TB

## Batch 1 (~60 minutes)
sbatch --array=1-162:9 --depend=afterok:95062 sip_2_procimg_array.sh
job 95076
Running data load: 3.12 TB

### Training 1 (~2-3 hours)
Data transfer takes most of the time.
sbatch --array=1-162:3 --depend=afterok:95076 sip_2_tune_train_array.sh
job 95078
Running data load: 3.01 TB

## Batch 2
sbatch --array=163-324:9 --depend=afterok:95078 sip_2_procimg_array.sh
job 95079

sbatch --array=163-324:1 --depend=afterok:95079 sip_2_tune_train_array.sh
job 95400

## Batch 3
sbatch --array=325-486:9 --depend=afterok:95400 sip_2_procimg_array.sh
job 97010
sbatch --array=325-486:1 --depend=afterok:97010 sip_2_tune_train_array.sh
job 97011

## Batch 4
sbatch --array=487-648:9 --depend=afterok:97011 sip_2_procimg_array.sh
job 97569
sbatch --array=487-648:1 --depend=afterok:97569 sip_2_tune_train_array.sh
job 97571

## Cleanup

There were 283 scenarios with results missing for at least one [feature x response variable]. Thus, I think I should rerun scenario 2.
sbatch --array=498-513:4 sip_2_procimg_array.sh
94449

sbatch --array=498-513:1 --depend=afterok:94449 sip_2_tune_train_array.sh
94454




