#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  20 16:21:10 2021

description: Configuration and functions for tropical cyclone (TC) model 
    comparison. You need so set some paths and parameters here before running 
    tc_comparison_main.py.
    Please refer to the paper (submitted) and README for more information.

@author: simonameiler
"""

import os
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
from climada.util.constants import SYSTEM_DIR # loads default directory paths for data
from climada.util.save import save
from climada.hazard import Centroids, TCTracks, TropCyclone
from climada.entity.exposures.litpop import LitPop
from climada.entity.exposures.base import INDICATOR_CENTR
from climada.engine import Impact
from climada.entity import IFTropCyclone, ImpactFuncSet
from climada.entity.impact_funcs.trop_cyclone import IFSTropCyclone
from climada.util.coordinates import dist_to_coast
from climada.util.coordinates import pts_to_raster_meta, get_resolution
        
#%% ################## Define local folders etc. ##############################
## define your local folders: (where to get data and save results)
# directory where TC tracks data is saved:
DATA_DIR = '/User/simonameiler/Documents/WCR/CLIMADA_DEVELOP/climada_python/data'
RES_DIR = os.path.join(DATA_DIR, 'results')

# countries by region:
region_ids_cal = {'NA1': ['AIA', 'ATG', 'ARG', 'ABW', 'BHS', 'BRB', 'BLZ', 
                          'BMU', 'BOL', 'CPV', 'CYM', 'CHL', 'COL', 'CRI', 
                          'CUB', 'DMA', 'DOM', 'ECU', 'SLV', 'FLK', 'GUF', 
                          'GRD', 'GLP', 'GTM', 'GUY', 'HTI', 'HND', 'JAM', 
                          'MTQ', 'MEX', 'MSR', 'NIC', 'PAN', 'PRY', 'PER', 
                          'PRI', 'SHN', 'KNA', 'LCA', 'VCT', 'SXM', 'SUR', 
                          'TTO', 'TCA', 'URY', 'VEN', 'VGB', 'VIR'], \
                  'NA2': ['CAN', 'USA'], \
                  'NI': ['AFG', 'ARM', 'AZE', 'BHR', 'BGD', 'BTN', 'DJI', 
                         'ERI', 'ETH', 'GEO', 'IND', 'IRN', 'IRQ', 'ISR', 
                         'JOR', 'KAZ', 'KWT', 'KGZ', 'LBN', 'MDV', 'MNG', 
                         'MMR', 'NPL', 'OMN', 'PAK', 'QAT', 'SAU', 'SOM', 
                         'LKA', 'SYR', 'TJK', 'TKM', 'UGA', 'ARE', 'UZB', 
                         'YEM'], \
                  'OC': ['ASM', 'AUS', 'COK', 'FJI', 'PYF', 'GUM', 'KIR', 
                         'MHL', 'FSM', 'NRU', 'NCL', 'NZL', 'NIU', 'NFK', 
                         'MNP', 'PLW', 'PNG', 'PCN', 'WSM', 'SLB', 'TLS', 
                         'TKL', 'TON', 'TUV', 'VUT', 'WLF'], \
                  'SI': ['COM', 'COD', 'SWZ', 'MDG', 'MWI', 'MLI', 'MUS', 
                         'MOZ', 'ZAF', 'TZA', 'ZWE'], \
                  'WP1': ['KHM', 'IDN', 'LAO', 'MYS', 'THA', 'VNM'], \
                  'WP2': ['PHL'], \
                  'WP3': ['CHN'], \
                  'WP4': ['HKG', 'JPN', 'KOR', 'MAC', 'TWN'], 
                  'ROW': ['ALB', 'DZA', 'AND', 'AGO', 'ATA', 'AUT', 'BLR', 
                          'BEL', 'BEN', 'BES', 'BIH', 'BWA', 'BVT', 'BRA', 
                          'IOT', 'BRN', 'BGR', 'BFA', 'BDI', 'CMR', 'CAF', 
                          'TCD', 'CXR', 'CCK', 'COG', 'HRV', 'CUW', 'CYP', 
                          'CZE', 'CIV', 'DNK', 'EGY', 'GNQ', 'EST', 'FRO', 
                          'FIN', 'FRA', 'ATF', 'GAB', 'GMB', 'DEU', 'GHA', 
                          'GIB', 'GRC', 'GRL', 'GGY', 'GIN', 'GNB', 'HMD', 
                          'VAT', 'HUN', 'ISL', 'IRL', 'IMN', 'ITA', 'JEY', 
                          'KEN', 'PRK', 'XKX', 'LVA', 'LSO', 'LBR', 'LBY', 
                          'LIE', 'LTU', 'LUX', 'MLT', 'MRT', 'MYT', 'MDA', 
                          'MCO', 'MNE', 'MAR', 'NAM', 'NLD', 'NER', 'NGA', 
                          'MKD', 'NOR', 'PSE', 'POL', 'PRT', 'ROU', 'RUS', 
                          'RWA', 'REU', 'BLM', 'MAF', 'SPM', 'SMR', 'STP', 
                          'SEN', 'SRB', 'SYC', 'SLE', 'SGP', 'SVK', 'SVN', 
                          'SGS', 'SSD', 'ESP', 'SDN', 'SJM', 'SWE', 'CHE', 
                          'TGO', 'TUN', 'TUR', 'UKR', 'GBR', 'UMI', 'ESH', 
                          'ZMB', 'ALA']}


# Define specific data for case study:
REGION = 'NA' # select 1 of the 4 regions to use here - only naming of files
BASIN = 'global' # IBTrACS ['NA', 'EP', 'NI', 'SI', 'WP', 'SP']
BASIN_K = 'AP' # basin selection Kerry ['AP', 'IO', 'SH', 'WP']
BASIN_S = ['EP','NA'] # str or list of basins for STORM tracks
HEMISPHERE = 'N'

# chose countries for exposure/centroids from Sam's calibration regions
cntry_list = []
reg_list = ['NA1','NA2']
for reg in reg_list:
    cntry_list.extend(region_ids_cal[reg])

ENTITY_DIR = os.path.join(DATA_DIR, 'entity') # where to save exposure data
ENTITY_STR = "litpop_%04das_%04d_%s.tif"
CENT_STR = "centroids_%04das_%s.hdf5"

# directory where TC tracks data is saved:
TRACKS_DIR = os.path.join(DATA_DIR, 'tracks')
KERRY_DIR = os.path.join(TRACKS_DIR, 'Kerry')
STORM_DIR = os.path.join(TRACKS_DIR, 'STORM', 'data')
CHAZ_DIR = os.path.join(TRACKS_DIR, 'CHAZ')
SYNTH_STR = "TC_tracks_synthetic_%04d-%04d_%s.p"
KERRY_STR= 'temp_%s_era5_reanalcal.mat'
CHAZ_STR = "TC_tracks_CHAZ_%s.p"
STORM_STR = "TC_tracks_STORM_%s.p"

HAZARD_DIR = os.path.join(DATA_DIR, 'hazard') # where to save hazard data
HAZARD_STR = "TC_%s_%04das_%s.hdf5" # filename for hazard data (will be saved in to this file when code is executed)

REF_YEAR = 2014 # reference year
RES_ARCSEC = 300 # resolution in arc seconds (best: 30)
YEAR_RANGE = [1980, 2018]

# boolean: load hazard & exposure data from HDF5 file after first creation? (speeds things up)
load_data_from_hdf5 = True

# set parallel processing
pool = Pool()

# boundaries of (sub-)basins (lonmin, lonmax, latmin, latmax)
BASIN_BOUNDS = {
    # North Atlantic/Eastern Pacific Basin
    'AP': [-120.0, 0.0, 0.0, 65.0],

    # Indian Ocean Basin
    'IO': [40.0, 100.0, 0.0, 40.0],

    # Southern Hemisphere Basin
    'SH': [0.0, 359.0, -60.0, 0.0],

    # Western Pacific Basin
    'WP': [90.0, -120.0, 0.0, 65.0],
}

# frequency corrections factors Kerry
freq_corr = {'AP': 9.4143, 'IO': 3.0734, 'SH': 4.4856, 'WP': 10.6551}
avg_freq = {'AP': 30.4, 'NA': 12.7, 'EP': 17.7, 'NI': 4.3, 'WP': 27.9, 'OC': 11.1, 'SI': 16.9, 'SH': 28}
dist_cst_lim = 1000000

#%%###################### Define functions ###################################

def init_coastal_litpop(countries, exp_dir, fn_str, res_arcsec = 300, \
                        ref_year=2014, dist_cst_lim=dist_cst_lim, lat_lim=70, \
                        save=True, region=REGION):
    
    """Initiates LitPop exposure of all provided countries within a defined 
    distance to coast and extent of lat, lon.

    Parameters:
        countries (list, optional): list with ISO3 names of countries, e.g
            ['ZWE', 'GBR', 'VNM', 'UZB']
        exp_dir (str):
        fn_str (str):
        ref_year (float):
        dist_cst_lim (float):
        lat_lim (float):
        save (boolean):
        region (str):
        
    Returns:
        DataFrame, hazard.centroids.centr.Centroids
    """
    
    success = []
    fail = []
    print("-----------------Initiating LitPop--------------------")
    exp_litpop = LitPop()
    print("Initiating LitPop country per country:....")
    for cntry in countries:
        print("-------------------------" + cntry + "--------------------------") 
        exp_litpop_tmp = LitPop()
        try:
            exp_litpop_tmp.set_country(cntry, res_arcsec=res_arcsec, reference_year=ref_year)
            exp_litpop_tmp.set_geometry_points()
            exp_litpop_tmp.set_lat_lon()
            try:
                reg_ids = np.unique(exp_litpop_tmp.region_id).tolist()
                dist_cst = dist_to_coast(np.array(exp_litpop_tmp.latitude), lon=np.array(exp_litpop_tmp.longitude))
                exp_litpop_tmp['dist_cst'] = dist_cst
                exp_litpop_tmp.loc[dist_cst > dist_cst_lim, 'region_id'] = -99
                exp_litpop_tmp = exp_litpop_tmp.loc[exp_litpop_tmp['region_id'].isin(reg_ids)]
                # exp_coast.plot_raster()
            except ValueError:
                print(cntry + ': distance to coast failed, exposure not trimmed')
            exp_litpop = exp_litpop.append(exp_litpop_tmp)
            success.append(cntry)
        except Exception as e:
            fail.append(cntry)
            print("Error while initiating LitPop Exposure for " + cntry + ". ", e)
    del exp_litpop_tmp
    print("----------------------Done---------------------")
    exp_litpop = exp_litpop.reset_index(drop=True)
    rows, cols, ras_trans = pts_to_raster_meta((exp_litpop.longitude.min(), \
            exp_litpop.latitude.min(), exp_litpop.longitude.max(), exp_litpop.latitude.max()), \
            min(get_resolution(exp_litpop.latitude, exp_litpop.longitude)))
    exp_litpop.meta = {'width':cols, 'height':rows, 'crs':exp_litpop.crs, 'transform':ras_trans}
    exp_litpop.set_geometry_points()
    exp_litpop.set_lat_lon()
    
    reg_ids = np.unique(exp_litpop.region_id).tolist()
    if -99 in reg_ids: reg_ids.remove(-99)
    if -77 in reg_ids: reg_ids.remove(-77)
    print('reg_ids:', reg_ids)
    exp_litpop.check()
    try:
        dist_cst = dist_to_coast(np.array(exp_litpop.latitude), lon=np.array(exp_litpop.longitude))
        print(max(dist_cst))
        exp_litpop['dist_cst'] = dist_cst
        exp_litpop.loc[dist_cst > dist_cst_lim, 'region_id'] = -99
        exp_litpop.loc[exp_litpop.latitude>lat_lim, 'region_id'] = -99
        exp_litpop.loc[exp_litpop.latitude<-lat_lim, 'region_id'] = -99
        print('rejected: ', np.argwhere(exp_litpop.region_id==-99).size)
        print('antes select:', exp_litpop.size)
        exp_coast = exp_litpop.loc[exp_litpop['region_id'].isin(reg_ids)]
        print('despues select:', exp_coast.size)

    except ValueError:
        print('distance to coast failed, exposure not trimmed')
        exp_coast = exp_litpop
    with open(os.path.join(exp_dir, 'cntry_fail.txt'), "w") as output:
        output.write(str(fail))
    with open(os.path.join(exp_dir, 'cntry_success.txt'), "w") as output:
        output.write(str(success))
    if save:
        exp_coast.write_hdf5(os.path.join(exp_dir, fn_str % (res_arcsec, ref_year, region)))
        print(os.path.join(exp_dir, fn_str % (res_arcsec, ref_year, region)))
    return exp_coast

def init_centroids_manual(bbox=[-66.5, 66.5, -179.5, 179.5], res_arcsec=3600, \
                           id_offset=1e9, on_land=False):
    """initiates centroids depeding on grid border points and resolution"""
    # number of centroids in lat and lon direction:
    n_lat = np.int(np.round((bbox[1]-bbox[0])*3600/res_arcsec))+1
    n_lon = np.int(np.round((bbox[3]-bbox[2])*3600/res_arcsec))+1
    
    cent = Centroids()
    mgrid= (np.mgrid[bbox[0] : bbox[1] : complex(0, n_lat), \
                           bbox[2] : bbox[3] : complex(0, n_lon)]). \
                  reshape(2, n_lat*n_lon).transpose()
    cent.set_lat_lon(mgrid[:,0], mgrid[:,1])
    cent.set_on_land()
    if not on_land: # remove centroids on land
        cent = cent.select(sel_cen=~cent.on_land)
    cent.set_region_id()
    cent.check()
    return cent

def init_exposure(countries,exp_dir, exp_fn_str, regs, make_plots=True, res_arcsec=300, ref_year=2014):

    if os.path.isfile(os.path.join(ENTITY_DIR, ENTITY_STR % (RES_ARCSEC, REF_YEAR, regs))):
        print("----------------------Loading Exposure----------------------")
        exp_coast = LitPop()
        exp_coast.read_hdf5(os.path.join(ENTITY_DIR, ENTITY_STR % (RES_ARCSEC, REF_YEAR, regs))) 
    else:
        print("----------------------Initiating Exposure-------------------")
        exp_coast = init_coastal_litpop(countries, exp_dir, exp_fn_str, \
                                            res_arcsec=res_arcsec, ref_year=ref_year,
                                            dist_cst_lim=dist_cst_lim, lat_lim=70)
    return exp_coast


def init_centroids(exp):
    cent = Centroids()
    if os.path.isfile(os.path.join(ENTITY_DIR, CENT_STR % (RES_ARCSEC, REGION))):
        print("----------------------Loading Exposure----------------------")
        cent.read_hdf5(os.path.join(ENTITY_DIR, CENT_STR % (RES_ARCSEC, REGION)))
    else:
        cent.set_lat_lon(np.array(exp.latitude), np.array(exp.longitude.values))
        exp[INDICATOR_CENTR] = np.arange(cent.lat.size)
        cent.region_id = np.array(exp.region_id.values, dtype='int64')
        cent.on_land = np.ones(cent.lat.size)
        cent_sea = init_centroids_manual(id_offset=10**(1+len(str(int(cent.size)))), \
                                          res_arcsec=3600)
        cent.append(cent_sea)
        if np.unique(cent.coord, axis=0).size != 2*cent.coord.shape[0]:
            cent.remove_duplicate_points()
        cent.check()
    #cent_.plot(c=cent.region_id)
    return cent

# Initiate TC tracks (IBTrACS)
def init_tc_tracks_IBTrACS():
    """initiate TC tracks from list of ibtracs IDs"""
    # initiate new instance of TCTracks class:
    tracks = TCTracks()
    # populate tracks by loading data from NetCDF:
    tracks.read_ibtracs_netcdf(year_range=YEAR_RANGE, correct_pres=False)
    tracks_IB = TCTracks()
    for i in range(0,6):
        filterdict = {'category': i}
        tracks_IB.data.extend(tracks.subset(filterdict).data)
    # post processing, increase time steps for smoother wind field:
    tracks_IB.equal_timestep(time_step_h=1., land_params=False)
    return tracks_IB


# Initiate hazard (for IBTrACS - Gettelman)
def init_tc_hazard(tracks, exposure, cent, key, load_haz=False):
    """initiate TC hazard from tracks and exposure"""
     # initiate new instance of TropCyclone(Hazard) class:
    tc_hazard = TropCyclone(pool)
    if load_haz and os.path.isfile(os.path.join(HAZARD_DIR, HAZARD_STR % (REGION, RES_ARCSEC, key))):
        print("----------------------Loading Hazard----------------------")
        tc_hazard.read_hdf5(os.path.join(HAZARD_DIR, HAZARD_STR % (REGION, RES_ARCSEC, key)))
    else:
        print("----------------------Initiating Hazard----------------------")
        # hazard is initiated from tracks, windfield computed:
        tc_hazard.set_from_tracks(tracks, centroids=cent)
        tc_hazard.check()
        tc_hazard.write_hdf5(os.path.join(HAZARD_DIR, HAZARD_STR % (REGION, RES_ARCSEC, key)))
    return tc_hazard

# Initiate TC tracks (IBTrACS probabilistic)
def calc_tracks(data_dir, basin):
    """ Generate synthetic tracks from ibtracs data, if not contained in data_dir.
    This functions is the longest one to execute."""
    try:
        abs_path = os.path.join(TRACKS_DIR, 'IBTrACS_p', SYNTH_STR %(YEAR_RANGE[0], YEAR_RANGE[1], basin))
        with open(abs_path, 'rb') as f:
            sel_ibtracs = pickle.load(f)
        print('Loaded synthetic tracks:', sel_ibtracs.size)
    except FileNotFoundError:
        # set parallel computing
        sel_ibtracs = TCTracks(pool)
        if basin == 'global':
            sel_ibtracs.read_ibtracs_netcdf(year_range=YEAR_RANGE, correct_pres=False)
            tracks_prob = TCTracks()
            for i in range(0,6):
                filterdict = {'category': i}
                tracks_prob.data.extend(sel_ibtracs.subset(filterdict).data)
        else:
            sel_ibtracs.read_ibtracs_netcdf(year_range=YEAR_RANGE, basin=basin, correct_pres=False)
            tracks_prob = TCTracks()
            for i in range(0,6):
                filterdict = {'category': i}
                tracks_prob.data.extend(sel_ibtracs.subset(filterdict).data)        
        tracks_prob.data = [x for x in tracks_prob.data if x.time.size > 1]
        tracks_prob.equal_timestep(time_step_h=1.)
        print('num tracks hist:', tracks_prob.size)
        tracks_prob.calc_random_walk()
        print('num tracks hist+syn:', tracks_prob.size)
        save(os.path.join(TRACKS_DIR,  'IBTrACS_p', SYNTH_STR %(YEAR_RANGE[0], YEAR_RANGE[1], basin)), tracks_prob)
    return tracks_prob

# Initiate CHAZ tracks
def init_CHAZ_tracks():
    """ Load all CHAZ tracks."""
    try:
        abs_path_CHAZ = os.path.join(CHAZ_DIR, CHAZ_STR %(REGION))
        with open(abs_path_CHAZ, 'rb') as f:
            tracks_CHAZ = pickle.load(f)
        print('Loaded CHAZ tracks:', tracks_CHAZ.size)
    except FileNotFoundError:
        ensembles = [[17], [7], [11], [15], [22], [38], [13], [8], [39], [19]]
        tracks_CHAZ = TCTracks(pool)
        for i_ens, ensemble in enumerate(ensembles):
           fname = os.path.join(CHAZ_DIR, f"global_new_00{i_ens}.nc")
           tr = TCTracks(pool)
           tr.read_simulations_chaz(fname, ensemble_nums=ensemble)
           tracks_CHAZ.append(tr.data)
        tracks_CHAZ.equal_timestep(time_step_h=1)
        save(os.path.join(CHAZ_DIR, CHAZ_STR %(REGION)), tracks_CHAZ)
    return tracks_CHAZ

def init_STORM_tracks(basin):
    """ Load all STORM tracks for the basin of interest."""
    try:
        abs_path_STORM = os.path.join(STORM_DIR, STORM_STR %(BASIN_K))
        with open(abs_path_STORM, 'rb') as f:
            tracks_STORM = pickle.load(f)
        print('Loaded STORM tracks:', tracks_STORM.size)
    except FileNotFoundError:
        # set parallel computing
        tracks_STORM = TCTracks(pool)
        all_tracks = []
        for j in basin:
            fname = lambda i: f"STORM_DATA_IBTRACS_{j}_1000_YEARS_{i}.txt"
            for i in range(10):
                tracks_STORM.read_simulations_storm(os.path.join(STORM_DIR, fname(i)))
                all_tracks.extend(tracks_STORM.data)
        tracks_STORM.data = all_tracks
        tracks_STORM.equal_timestep(time_step_h=1.)
        save(os.path.join(STORM_DIR, STORM_STR %(BASIN_K)), tracks_STORM)
    return tracks_STORM

