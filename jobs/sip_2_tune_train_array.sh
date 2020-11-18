#!/bin/bash -l
#SBATCH --array=-1
#SBATCH --job-name=sip_2_train_array
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem-per-cpu=2gb
#SBATCH -t 4:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nigo0024@umn.edu
#SBATCH -o ./reports_tunetrain_array/R-%A-%a-%x.out
#SBATCH -e ./reports_tunetrain_array/R-%A-%a-%x.err

((start_n=$SLURM_ARRAY_TASK_ID))
((n=$SLURM_ARRAY_TASK_STEP))
((msi_run_id=2))
start=$(($start_n-1))

# cd ../../.. $SLURM_SUBMIT_DIR
cd /panfs/roc/groups/5/yangc1/public/hs_process
CID="5d573462-2134-4900-970d-6e7a5e0f2b1e"
TT="AgBXVNMKXoOKa6XBlympD0pVKq3EkXxl03NkGB56YPqweBayGeFyClPv0n86GkyPK7PP0mgNM4GqNCk32vwoclEEN"
TRT="AgxdYn0G06XqDYKmy21kpjmr7xkoP3kdBdrVzPOYag6z3vXp6eCeUNX4Pb4qwa4x9oW5O8DPo3KN684yoNKp05pKrGmyQ"
export msi_run_id start n CID TT TRT

conda activate sip_run_2

python sip/tune_train.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $start+0 --idx_max $start+$n --CLIENT_ID $CID --TRANSFER_TOKEN $TT --TRANSFER_REFRESH_TOKEN $TRT --feat reflectance --y_label biomass_kgha
python sip/tune_train.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $start+0 --idx_max $start+$n --CLIENT_ID $CID --TRANSFER_TOKEN $TT --TRANSFER_REFRESH_TOKEN $TRT --feat reflectance --y_label nup_kgha
python sip/tune_train.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $start+0 --idx_max $start+$n --CLIENT_ID $CID --TRANSFER_TOKEN $TT --TRANSFER_REFRESH_TOKEN $TRT --feat reflectance --y_label tissue_n_pct
python sip/tune_train.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $start+0 --idx_max $start+$n --CLIENT_ID $CID --TRANSFER_TOKEN $TT --TRANSFER_REFRESH_TOKEN $TRT --feat derivative_1 --y_label biomass_kgha
python sip/tune_train.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $start+0 --idx_max $start+$n --CLIENT_ID $CID --TRANSFER_TOKEN $TT --TRANSFER_REFRESH_TOKEN $TRT --feat derivative_1 --y_label nup_kgha
python sip/tune_train.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $start+0 --idx_max $start+$n --CLIENT_ID $CID --TRANSFER_TOKEN $TT --TRANSFER_REFRESH_TOKEN $TRT --feat derivative_1 --y_label tissue_n_pct
python sip/tune_train.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $start+0 --idx_max $start+$n --CLIENT_ID $CID --TRANSFER_TOKEN $TT --TRANSFER_REFRESH_TOKEN $TRT --feat derivative_2 --y_label biomass_kgha
python sip/tune_train.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $start+0 --idx_max $start+$n --CLIENT_ID $CID --TRANSFER_TOKEN $TT --TRANSFER_REFRESH_TOKEN $TRT --feat derivative_2 --y_label nup_kgha
python sip/tune_train.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $start+0 --idx_max $start+$n --CLIENT_ID $CID --TRANSFER_TOKEN $TT --TRANSFER_REFRESH_TOKEN $TRT --feat derivative_2 --y_label tissue_n_pct
wait

for ((i=0; i < $n; i++))
do python sip/transfer_data_level.py --CLIENT_ID $CID --TRANSFER_TOKEN $TT --TRANSFER_REFRESH_TOKEN $TRT --msi_run_id $msi_run_id --idx_grid $start+$i --level "segment"
done
