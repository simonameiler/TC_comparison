#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  20 16:21:10 2021

description: Main analyses for tropical cyclone (TC) model comparison.
    Please refer to the paper (submitted) and README for more information.

@author: simonameiler
"""

import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy as cp
from pathlib import Path
from pathos.pools import ProcessPool as Pool

import sys
#sys.path.append('/cluster/work/climate/meilers/climada_python')

# import CLIMADA modules:
#from climada.util.constants import SYSTEM_DIR # loads default directory paths for data
from climada.util.save import save
from climada.hazard import Centroids, TCTracks, TropCyclone
from climada.entity.exposures.litpop import LitPop
from climada.entity.exposures.base import INDICATOR_CENTR
from climada.engine import Impact
from climada.entity import IFTropCyclone, ImpactFuncSet
from climada.entity.impact_funcs.trop_cyclone import IFSTropCyclone
from climada.util.coordinates import dist_to_coast
from climada.util.coordinates import pts_to_raster_meta, get_resolution
from TC_comparison_functions import *

#%% Calling functions

# exposure - the exosure reads litpop exposure for a list of countries.
exposure = init_exposure(cntry_list, make_plots=False, res_arcsec=RES_ARCSEC, ref_year=REF_YEAR)
   
exposure.plot_raster()
# exposures data. --> Set impact function ID in exposures
cent = init_centroids(exposure)

# write TC tracks into dictionary
tracks_dict = dict()
# tracks_IBTrACS = init_tc_tracks_IBTrACS()
# tracks_IBTrACS_prob = calc_tracks(DATA_DIR, BASIN) # calculate probabilistic tracks
# tracks_dict['IBTrACS'] = tracks_IBTrACS
# tracks_dict['IBTrACS_p'] = tracks_IBTrACS_prob

# tracks_Kerry = TCTracks()
# tracks_Kerry.read_simulations_emanuel(os.path.join(KERRY_DIR, KERRY_STR % (BASIN_K)), hemisphere=HEMISPHERE)
# tracks_Kerry.equal_timestep(time_step_h=1.)
# tracks_dict['Kerry_'+str(BASIN_K)] = tracks_Kerry

# tracks_CHAZ = init_CHAZ_tracks()
# tracks_dict['CHAZ'] = tracks_CHAZ

# # possible to read in STORM tracks for multiple basins
# tracks_STORM = init_STORM_tracks(basin=BASIN_S)
# tracks_dict['STORM'] = tracks_STORM

tracks_dict = ['IBTrACS', 'IBTrACS_p', 'STORM', 'Kerry_'+str(BASIN_K), 'CHAZ']

# create hazard dictionary
hazard_dict = dict()
for tr in tracks_dict:
    #hazard = init_tc_hazard(tracks_dict[tr], exposure, cent, tr, load_haz=True)
    hazard = init_tc_hazard(tr, exposure, cent, tr, load_haz=True)
    hazard_dict[tr] = hazard

# hazard frequency corrections
# STORM
freq_corr_STORM = 1/10000
#freq_corr_STORM = 1.4615/10000
hazard_dict['STORM'].frequency = hazard_dict['STORM'].frequency*freq_corr_STORM

# Kerry
hazard_dict['Kerry_'+str(BASIN_K)].frequency = np.ones(hazard_dict['Kerry_'+str(BASIN_K)].event_id.size)*freq_corr[BASIN_K]/hazard_dict['Kerry_'+str(BASIN_K)].size

# CHAZ
freq_corr_CHAZ = 1/320
#hazard_dict['CHAZ'].frequency = np.ones(hazard_dict['CHAZ'].event_id.size)*avg_freq[REGION]/hazard_dict['CHAZ'].event_id.size
hazard_dict['CHAZ'].frequency = np.ones(hazard_dict['CHAZ'].event_id.size)*freq_corr_CHAZ

# The iso3n codes need to be consistent with the column “region_id” in the 
# 1. Init impact functions:
impact_func_set = IFSTropCyclone()
impact_func_set.set_calibrated_regional_IFs(calibration_approach='TDR') 
# get mapping: country ISO3n per region:
iso3n_per_region = impf_id_per_region = IFSTropCyclone.get_countries_per_region()[2]

code_regions = {'NA1': 1, 'NA2': 2, 'NI': 3, 'OC': 4, 'SI': 5, 'WP1': 6, \
                'WP2': 7, 'WP3': 8, 'WP4': 9, 'ROW': 10}

# match exposure with correspoding impact function
for calibration_region in impf_id_per_region:
    for country_iso3n in iso3n_per_region[calibration_region]:
        exposure.loc[exposure.region_id== country_iso3n, 'if_TC'] = code_regions[calibration_region]
        exposure.loc[exposure.region_id== country_iso3n, 'if_'] = code_regions[calibration_region]

# finally… impact is calculated:
impact_dict = dict()
for imp in hazard_dict:
    impact = Impact()
    #exposure.assign_centroids(imp)
    impact.calc(exposure, impact_func_set, hazard_dict[imp], save_mat=True)
    impact_dict[imp] = impact

#%% Exceedance probability curve and annual average impact

freq_curves = dict()
aai_agg = dict()

for freq in impact_dict:
    freq_curves[freq] = impact_dict[freq].calc_freq_curve()
    aai_agg[freq] = impact_dict[freq].aai_agg
 
for name, aai in aai_agg.items():
    print('{:<30} {:<15}'.format(name, aai))
    #print(aai/exp_coast.value.sum())
save_table_str = "aai_agg_%s_%04das_%04d-%04d_ERA5.csv"
csv_file = os.path.join(RES_DIR, save_table_str %(REGION, RES_ARCSEC, YEAR_RANGE[0], YEAR_RANGE[1]))
with open(csv_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for data in aai_agg.items():
        writer.writerow(data)

# plot exceedance frequency curves for all tracks - log scale    
fig, axis = plt.subplots(1,1, figsize=(6,4), sharex=True, sharey=True, tight_layout=False)
for plots in freq_curves:
    freq_curves[plots].plot(axis,log_frequency=False,linestyle='solid',label=plots)
axis.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
plt.xscale('log')

save_fig1_str = "exceedance_freq_curve_TC_%s_%04das.png"
plt.savefig(os.path.join(RES_DIR, save_fig1_str %(REGION, RES_ARCSEC)), \
    dpi=300, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format='png', \
        ransparent=False, bbox_inches='tight', pad_inches=0.1, \
        frameon=None, metadata=None)

    
# exceedance frequency curve for RP=250
freq_curves_RP250 = dict()
rp = np.arange(0,250)

for freq in impact_dict:
    freq_curves_RP250[freq] = impact_dict[freq].calc_freq_curve(rp)
# plot exceedance frequency curves for all tracks - log scale    
fig, axis = plt.subplots(1,1, figsize=(6,4), sharex=True, sharey=True, tight_layout=False)
for plots in freq_curves_RP250:
    freq_curves_RP250[plots].plot(axis,log_frequency=False,linestyle='solid',label=plots)
axis.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
plt.xscale('log')

save_fig1_str = "exceedance_freq_curve_TC_RP250_%s_%04das.png"
plt.savefig(os.path.join(RES_DIR, save_fig1_str %(REGION, RES_ARCSEC)), \
    dpi=300, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format='png', \
        ransparent=False, bbox_inches='tight', pad_inches=0.1, \
        frameon=None, metadata=None)
    
# hazard intensity histogram
max_haz = dict()
for haz in hazard_dict:
    #max_haz[haz] = hazard_dict[haz].intensity.max(axis=1)    
    max_haz[haz] = hazard_dict[haz].intensity[:,cent.lat[cent.on_land>0]].max(axis=1)
    
n_bins = [15,25,35,45,55,65,75,85,95]    
    
fig, ax = plt.subplots(1, 1, figsize=(4.5,4), sharex=True, sharey=True, tight_layout=True)
x0 = ['IBTrACS', 'IBTrACS_p', 'STORM', 'Kerry_'+str(BASIN_K), 'CHAZ']
x0_multi = [max_haz[haz].data for haz in x0]
ax.hist(x0_multi, bins=n_bins, density=True, histtype='bar', label=x0)
ax.legend(loc='center right', bbox_to_anchor=(1, 0.85))
fig.text(0.5, 0, 'max wind intensity', ha='center',fontsize=plt.rcParams['axes.labelsize'])
fig.text(0, 0.5, 'frequency', va='center', rotation='vertical',fontsize=plt.rcParams['axes.labelsize'])

save_fig2_str = "his_max_haz_TC_%s_%04das.png"
plt.savefig(os.path.join(RES_DIR, save_fig2_str %(REGION, RES_ARCSEC)), \
    dpi=300, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format='png', \
        ransparent=False, bbox_inches='tight', pad_inches=0.1, \
        frameon=None, metadata=None)    
    
#%% Probability density analysis - analogous to Kerry

# calculate impact year sets and write values into list - if log is
annual_damage = list(impact_dict['IBTrACS'].calc_impact_year_set().values())
# convert to pandas Series
annual_damage_series = pd.Series(annual_damage)
# plot pdf from series (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.density.html)
# Generate Kernel Density Estimate plot using Gaussian kernels.
annual_damage_series.plot.kde()

#%% Probability density analysis - does not make sense...I think (the at event perspective)

# calculate impact AT EVENT and write values into list - if log is
annual_damage = list(impact_dict['IBTrACS'].at_event)
# convert to pandas Series
annual_damage_series = pd.Series(annual_damage)
# plot pdf from series (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.density.html)
# Generate Kernel Density Estimate plot using Gaussian kernels.
annual_damage_series.plot.kde()

#%% 
mask = Centroids()
if os.path.isfile(os.path.join(ENTITY_DIR, CENT_STR % (RES_ARCSEC, REGION))):
    print("----------------------Loading Exposure----------------------")
    mask.read_hdf5(os.path.join(ENTITY_DIR, CENT_STR % (RES_ARCSEC, REGION)))
else:
    mask.set_lat_lon(np.array(exposure.latitude), np.array(exposure.longitude.values))
    exposure[INDICATOR_CENTR] = np.arange(mask.lat.size)
    mask.region_id = np.array(exposure.region_id.values, dtype='int64')
    mask.set_on_land = np.ones(mask.lat.size)
    
cent.lat[cent.on_land>0].size
