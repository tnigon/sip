# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:00:07 2020

@author: nigo0024
"""
import os
import subprocess

msi_dir = 'results/ref_closest_panel/crop_plot/clip_none/smooth_none/seg_mcari2_50_upper/'
tier2_dir = os.path.join('S3://', msi_dir)
# tier2_dir = 'S3://results/ref_closest_panel/crop_plot/clip_none/smooth_none'

# s3cmd put --recursive dir1 s3://s3tools-demo/some/path/
subprocess.call(['s3cmd', 'put', '-r', msi_dir, tier2_dir])
subprocess.call(['rm', '-r', msi_dir])
# rm -r results/ref_closest_panel/crop_plot/clip_none/smooth_none/seg_mcari2_90_upper