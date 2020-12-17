# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:01:37 2020

@author: nigo0024
"""
import spotpy
import numpy as np

class spotpy_setup_sip(object):
    def __init__(self):
        # self.params = [spotpy.parameter.List('dir_panels',['closest', 'all']),
        #                spotpy.parameter.List('crop',['plot_bounds', 'buffer']),
        #                spotpy.parameter.List('clip',['none', 'ends', 'all']),
        #                spotpy.parameter.List('smooth',['none', 'sg-11']),
        #                spotpy.parameter.List('bin',['none', 'sentinel-2a_mimic', 'bin_20nm']),
        #                spotpy.parameter.List('segment',
        #                                      ['none', 'ndi_upper_50', 'ndi_lower_50', 'mcari2_upper_50',
        #                                       'mcari2_lower_50', 'mcari2_upper_90', 'mcari2_in_50-75',
        #                                       'mcari2_in_75-90', 'mcari2_upper_90_green_upper_75'])]
        self.params = [spotpy.parameter.List('dir_panels',[0, 1], repeat=True),
                       spotpy.parameter.List('crop',[0, 1], repeat=True),
                       spotpy.parameter.List('clip',[0, 1, 2], repeat=True),
                       spotpy.parameter.List('smooth',[0, 1], repeat=True),
                       spotpy.parameter.List('bin',[0, 1, 2], repeat=True),
                       spotpy.parameter.List('segment',[0, 1, 2, 3, 4, 5, 6, 7, 8], repeat=True)]
    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self,vector):
        x=np.array(vector)
        simulations = [sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)]
        return simulations

    def evaluation(self):
        observations=[0]
        return observations

    def objectivefunction(self,simulation,evaluation):
        # objectivefunction=-spotpy.objectivefunctions.rmse(evaluation,simulation)
        objectivefunction = 1
        return objectivefunction