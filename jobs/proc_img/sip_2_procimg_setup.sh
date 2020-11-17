#!/bin/bash -l
#SBATCH --array=-1
#SBATCH --job-name=sip_2_procimg_setup
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem-per-cpu=2gb
#SBATCH -t 0:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nigo0024@umn.edu
#SBATCH -o ./reports/R-%A-%a-%x.out
#SBATCH -e ./reports/R-%A-%a-%x.err

((start_n=$SLURM_ARRAY_TASK_ID))
# Set n=1 instead of n=$SLURM_ARRAY_TASK_STEP
((n=1))
((msi_run_id=2))
start=$(($start_n-1))

# Run for every 27th scenario, but just 2 times (covers smooth scenarios)
((seg_n=9))
((bin_n=3))

# cd ../../.. $SLURM_SUBMIT_DIR
cd /panfs/roc/groups/5/yangc1/public/hs_process
CID="5d573462-2134-4900-970d-6e7a5e0f2b1e"
TT="AgBXVNMKXoOKa6XBlympD0pVKq3EkXxl03NkGB56YPqweBayGeFyClPv0n86GkyPK7PP0mgNM4GqNCk32vwoclEEN"
TRT="AgxdYn0G06XqDYKmy21kpjmr7xkoP3kdBdrVzPOYag6z3vXp6eCeUNX4Pb4qwa4x9oW5O8DPo3KN684yoNKp05pKrGmyQ"
export msi_run_id start n seg_n bin_n CLIENT_ID TT TRT

conda activate sip_run_2

python sip/process_img.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $((start+$((seg_n*bin_n*0)))) --idx_max $((start+$((seg_n*bin_n*0))))+$n
python sip/process_img.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $((start+$((seg_n*bin_n*1)))) --idx_max $((start+$((seg_n*bin_n*0))))+$n
wait
