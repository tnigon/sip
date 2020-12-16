# Run 2 slurm commands

The following commands were passed to Mangi (amdsmall). The job ID of each is recorded.

Run date: 2020-11-20

Running data load: 0.15 TB

## Setup (~20 minutes)
sbatch --array=1-648:54 sip_2_procimg_setup.sh
job 293506
Running data load: 0.78 TB

## Batch 1 (~60 minutes)
sbatch --array=1-162:9 --depend=afterok:293506 sip_2_procimg_array.sh
job 293527
Running data load: 3.12 TB

### Training 1 (~2-3 hours)
Data transfer takes most of the time.
sbatch --array=1-162:3 --depend=afterok:293527 sip_2_tune_train_array.sh
job 293545
Running data load: 3.01 TB

## Batch 2
sbatch --array=163-324:9 --depend=afterok:293545 sip_2_procimg_array.sh
job 294849

sbatch --array=163-324:1 --depend=afterok:294849 sip_2_tune_train_array.sh
job 294850

## Batch 3
sbatch --array=325-486:9 --depend=afterok:294850 sip_2_procimg_array.sh
job 296123
sbatch --array=325-486:1 --depend=afterok:296123 sip_2_tune_train_array.sh
job 296124

## Batch 4
sbatch --array=487-648:9 --depend=afterok:296124 sip_2_procimg_array.sh
job 297114
sbatch --array=487-648:1 --depend=afterok:297114 sip_2_tune_train_array.sh
job 297119

## Cleanup

There were 5 scenarios with results missing for at least one [feature x response variable]. Trying them again.
sbatch --array=482,454,192,169,369:1 sip_2_procimg_array.sh



There were 283 scenarios with results missing for at least one [feature x response variable]. Thus, I think I should rerun scenario 2.
sbatch --array=498-513:4 sip_2_procimg_array.sh
94449

sbatch --array=498-513:1 --depend=afterok:94449 sip_2_tune_train_array.sh
94454




