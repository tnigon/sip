# Run 2 slurm commands

The following commands were passed to Mangi (amdsmall). The job ID of each is recorded.

Run date: 2020-11-18

Running data load: 0.15 TB

sbatch --array=55-108:54 sip_2_procimg_setup.sh
82100
sbatch --array=55-108:3 --depend=afterok:82100 sip_2_tune_train_array.sh
82101

## Setup (~25 minutes)
sbatch --array=1-648:54 sip_2_procimg_setup.sh
job 81388
Running data load: 0.69 TB

## Batch 1
sbatch --array=1-162:9 --depend=afterok:81388 sip_2_procimg_array.sh
job 81404
Running data load: 2.29 TB

sbatch --array=1-162:3 --depend=afterok:81404 sip_2_tune_train_array.sh
job 81435

## Batch 2
sbatch --array=163-324:9 --depend=afterok:81435 sip_2_procimg_array.sh
job 81
sbatch --array=163-324:3 --depend=afterok:81 sip_2_tune_train_array.sh
job 81

## Batch 3
sbatch --array=325-486:9 --depend=afterok:81 sip_2_procimg_array.sh
job 81
sbatch --array=325-486:3 --depend=afterok:81 sip_2_tune_train_array.sh
job 81

## Batch 4
sbatch --array=487-648:9 --depend=afterok:81 sip_2_procimg_array.sh
job 81
sbatch --array=487-648:3 --depend=afterok:81 sip_2_tune_train_array.sh
job 81