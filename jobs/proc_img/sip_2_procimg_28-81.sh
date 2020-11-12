#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem-per-cpu=2580mb
#SBATCH -t 4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nigo0024@umn.edu
#SBATCH --job-name=msi_2_procimg_28-81
#SBATCH -o R-%j-%x.out
#SBATCH -e R-%j-%x.err

cd $SLURM_SUBMIT_DIR
((msi_run_id=2))
((start=27))
CLIENT_ID="5d573462-2134-4900-970d-6e7a5e0f2b1e"
TRANSFER_TOKEN="AgBXVNMKXoOKa6XBlympD0pVKq3EkXxl03NkGB56YPqweBayGeFyClPv0n86GkyPK7PP0mgNM4GqNCk32vwoclEEN"
TRANSFER_REFRESH_TOKEN="AgxdYn0G06XqDYKmy21kpjmr7xkoP3kdBdrVzPOYag6z3vXp6eCeUNX4Pb4qwa4x9oW5O8DPo3KN684yoNKp05pKrGmyQ"
export msi_run_id start CLIENT_ID TRANSFER_TOKEN TRANSFER_REFRESH_TOKEN

conda activate sip_run_2

python sip/process_img.py --n_jobs $SLURM_NTASKS --msi_run_id $msi_run_id --idx_min $start+0 --idx_max $start+54
wait
