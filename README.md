# Subjective Image Processing
The purpose of this repository is to answer the question: What processing steps were more influential than others in the final modeling results?

## Environment
`conda create -n msi_sip python=3.7 -y`
`conda config --env --add channels conda-forge`
`conda config --env --set channel_priority strict`
# `conda install python=3 geopandas -y`

`conda install -c conda-forge hs-process`
`conda install -c conda-forge scikit-learn`
`conda install -c conda-forge globus-sdk`
`conda install -c conda-forge tqdm`
`conda install -c conda-forge boto3`
`pip install git+https://github.com/tnigon/hs_process.git@dev`


## Run a job on MSI
- login via PuTTY
- `ssh mangi`
- `cd ../public/hs_process`
- `conda activate msi_sip_37`
- `qsub <job_file.pbs>`
More information at [MSI Job Submission and Scheduling (PBS Scripts)](https://www.msi.umn.edu/content/job-submission-and-scheduling-pbs-scripts)
# To Do

- Function to move intermediate data and results from MSI high performance to second tier storage
  - tier2_data_transfer(): transfers the segmented data at teh bottom of the "data" directory
  - tier2_results_transfer(): transfers the "msi_0_###" results folder at the end of the loop for that run_id (run_id expresses as ###)
--Run your python script using .pbs script and qsub to submit the job on a single core
  -NOT PARALLEL
  -Same as running on your desktop
- Rewrite hs_script.py function hs_grid_search in .pbs script using

# To Do [completed]
- Transfer all data to MSI account using GLOBUS (follow MSI instructions)


# Compute Requirements
- How long is model training taking? Would shifting scikitlearn to GPU increase speed?
- Total disk space required for naive approach is 12.1TB

# Description of Input Data

## Image pre-processing
Before images were uploaded to MSI high performance directory, the following pre-processing steps were performed:
1. Images converted from raw (digital number) to radiance using calibration file provided by Resonon.
2. Images georectified (with GPS and IMU data to project each image line to a spatial reference system)
3. Images converted from radiance to reflectance using measured spectra from reference panesl and the radiance from the reference panels (from images) -- possible paper: find reference panels autonomously in images.

## Image post-processing
1. Images cropped (either to plot boundaries or with an added negative buffer to remove boundary plants).
2. [hs_process](https://hs-process.readthedocs.io/en/latest/) library is being used to achieve all image post-processing steps for this project (carried out by the script).

## Data on MSI
Hyperspectral reflectance images that have undergone the cropping step were uploaded to MSI. These images are named with the following unique identifiers in their filename:
- study name (aerfsmall, aerfwhole, or wells)
- acquisition date
- plot_id

## Image file format
- .bip: hyperspectral datacubes and band math images (single band)
- .bip.hdr: header file (see the [ENVI .hdr specification](http://www.harrisgeospatial.com/docs/enviheaderfiles.html) for more information)
- .hdr "history" tag shows all operation done to an image up to this point (follows the convention of Spectranon (Resonon software)).
- .spec: BIP file with only a single "pixel"; .spec files contain data across the full spctral domain and do not contain any spatial information (they do have an accomanying .hdr file)

## MSI Folder structure
At the highest level, there are two directories:
1. data: contains the ground truth data, as well as all the image/spectral data before and after MSI processing.
2. results: contains all the tuning, training, and testing results of the supervised regression models (Lasso and partial least squares regression used currently)

\hs_process
  \data
  ...
    \ref_all_panels {images where the average of all panels used to convert to reflectance}
    \ref_closest_panel {images where the radiance from the closest panel in time was used to covert to reflectance}
      ...
      \crop_buf {cropped images with a negative buffer so the sampling extent roughly matches the imaging extent}
      \crop_plot {cropped images by the extent of the plot boundary}
        ...
        \clip_all {images with O2 absorption, H2O absorption, and end bands clipped from the spectra}
        \clip_ends {end bands clipped from spectra}
        \clip_none {no bands clipped from spectra}
          ...
          \smooth_none {images without any pixel-wise smoothing across the spectral domain}
          \smooth_window_5 {pixels smoothed across the spectral domain using Sovitzky-Golay smoothing with window size of 5}
          \smooth_window_11 {Sovitzky-Golay smoothing with window size of 5}
            ...
            \bm_green {green band math images (green reflectance used as an indicator for segmentation)}
            \bm_mcari2 {MCARI2 band math images}
            \bm_nir {NIR band math images}
            \seg_mcari2_50_75_between {segmentation results with pixels below 50th pctl and above 75th pctl MCARI2 masked out}
            \seg_mcari2_50_upper {segmentation results with pixels below 50th pctl MCARI2 masked out}
            \seg_mcari2_75_95_between {segmentation results with pixels below 75th pctl and above 95th pctl MCARI2 masked out}
            \seg_mcari2_90_upper {segmentation results with pixels below 90th pctl MCARI2 masked out}
            \seg_mcari2_90_upper_green_75_upper {segmentation results with pixels below 90th pctl MCARI2 and below 75th pctl green masked out}
            \seg_mcari2_90_upper_nir_75_upper {segmentation results with pixels below 90th pctl MCARI2 and below 75th pctl NIR masked out}
            \seg_mcari2_95_98_between {segmentation results with pixels below 95th pctl and above 98th pctl MCARI2 masked out}
            \seg_none {results with no pixels masked out}
            ...
  \results
    \msi_0_results {where "0" represents the MSI run ID}
      \msi_0_000 {where "000" represents the processing scenario}
      \msi_0_001
      ...
      \msi_0_288
        \biomass_kgha {for each ground truth measure we are trying to predict}
        \nup_kgha
        \tissue_n_pct
          \aux_mcari2_pctl_10th {spectral features plus the 10th percentile MCARI2 feature}
          \spectral {spectral features only}
            \testing {test results}
              \figures {figures plotting measured vs. predicted, as well as train/test error as a function of feature number}
            \tuning {tuning results}

## hs_script.py
This file contains the main loop to carry out all image processing tasks, all model tuning, training, and testing, as well as creation of figures/plots of final data.

### Image processing loop
- clip -- clip()
- smooth -- smooth()
- segmentation -- seg()
This order matches the folder structure, and a new folder is created for each possible image procesing scenario. Folder hierarchy gets created as the loop progresses according to the hs_settings in "hs_script.py". Processed images are saved in the appropriate folder directory.

### Hyperparameter tuning loop
Tuning and all subsequent steps are carried out separately for each ground truth measurement, as well as each set of available features to be evaluated.
Ground truth measurements:
- above-ground biomass (kg ha-1)
- total nitrogen uptake (kg ha-1)
- tissue nitrogen concenteration (%)
Available features to consider:
- spectral features only
- spectral features plus the 10th percentile MCARI2 value
Supervised regression models evaluated:
- Lasso regression
- Partial least squares regression
Hyperparameter tuning is carried out by splitting the training dataset (60% of samples) using a repeated stratified k-fold cross validation (4 splits and 3 replications). Thus, each tuning fold uses 75% of the training samples, which use 60% of the total samples.
The results of hyperparameter tuning are saved to the "tuning" folder in the appropriate directory

### Model training and testing
Following hyperparameter tuning, each model is trained on the full training set (60% of all samples) using the optimal hyperparameters deterimined from the tuning step. The trained model is then used to predict each ground truth measurement using the test set (40% of samples). Test predictions and scores are saved to a .csv for each ground truth measurement and model. Figures are also created that show measured vs. predicted values for each number of features, as well as a figure that shows error as a function of feature number.

### Other tasks
- The time it takes to execute each step in the loop is recorded and saved to a .csv file in the base "results" folder ("msi_0_runtime.csv")
- A .csv is created for each ground truth sample and each error metric (MAE, MSE, and R2) and is also saved in the base "results" folder (e.g., "msi_0_biomass_kgha_MAE.csv")
