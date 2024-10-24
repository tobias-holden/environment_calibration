import sys
sys.path.append('../')
import numpy as np
import warnings
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'
import os
import manifest as manifest
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from datetime import timedelta
from helpers import load_coordinator_df

sns.set_context("talk")
sns.set_style("white")

from compare_to_data.calculate_all_scores import  compute_all_scores, load_case_data

def compute_scores_across_site(site):
    coord_df = load_coordinator_df()
    scores = compute_all_scores(site)
    #print(scores)
    weights = pd.read_csv(os.path.join(manifest.input_files_path,"my_weights.csv"),index_col=0)
    #print(weights)
    scores['eir_score'] = [float(weights.at['eir_score','weight'])*val for val in scores['eir_score']]
    if(coord_df.at['incidence_comparison','value']):
        scores['shape_score'] = [float(weights.at['shape_score','weight'])*val for val in scores['shape_score']]
        scores['intensity_score'] = [float(weights.at['intensity_score','weight'])*val for val in scores['intensity_score']]
    if(coord_df.at['prevalence_comparison','value']):
        scores['prevalence_score'] = [float(weights.at['prevalence_score','weight'])*val for val in scores['prevalence_score']]
    
    scores = scores.rename(columns={"Sample_ID":"param_set"})
    scores = pd.melt(scores, id_vars="param_set")
    scores = scores.groupby("param_set")['value'].agg('sum').reset_index(name='score')
    
    return scores

def plot_incidence(site="",plt_dir=os.path.join(manifest.simulation_output_filepath,"_plots"),wdir='.',agebin=5,start_year=0):
    
    best = pd.read_csv(os.path.join(wdir,"emod.best.csv"))
    best = best['param_set'][0]
    
    case_df = load_case_data()
    # filter to DS_Name
    case_df = case_df[case_df['DS_Name']==site]
    # filter to age ov5
    case_df = case_df[case_df['age']=='ov5']
    # convert case_df 'year' to start at 0, like simulations
    case_df['year'] = [(y - start_year) for y in case_df['year']]
    # sum incidence across year    
    case_df=case_df.merge(case_df.groupby('year')['repincd'].agg(np.nanmax).reset_index(name='max_incd'), on='year',how='left')
    # normalize monthly incidence so each year sums to 1
    case_df['norm_repincd'] = case_df.apply(lambda row: (row['repincd'] / row['max_incd']), axis=1)
    # get average normalized incidence per month
    rcases = case_df.groupby(['month','DS_Name'])['norm_repincd'].agg(np.nanmean).reset_index()
    
    sim_cases = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"PfPR_ClinicalIncidence_monthly.csv"))
    # filter to age 5+
    sim_cases = sim_cases[sim_cases['agebin']==agebin]
    # get mean population and clinical cases by month, year, and Sample_ID
    sim_cases['Inc'] = sim_cases['Cases'] / sim_cases['Pop']
    sim_cases=sim_cases.merge(sim_cases.groupby(['Sample_ID','Year'])['Inc'].agg(np.nanmax).reset_index(name='max_simincd'), on=['Sample_ID'],how='left').reset_index()
    sim_cases['norm_simincd'] = sim_cases.apply(lambda row: (row['Inc'] / row['max_simincd']), axis=1)
    # get mean normalized incidence by month/sample_id (across years)
    sim_cases = sim_cases.groupby(['Sample_ID', 'month','agebin'])['norm_simincd'].agg(np.nanmean).reset_index()
    # merge simulated normalized monthly incidence with reference data on ['month']
    score1 = sim_cases.merge(rcases, on ='month')
    score1 = score1.dropna(subset=['norm_simincd']).reset_index()
    print(score1)
    print(best)
    print(score1['Sample_ID'].unique())
    score1 = score1[score1['Sample_ID']==best]
    #print(rcases1)
    #print(sim_cases1)
    print(score1)
    plt.figure(3, figsize=(6, 6), dpi=300, tight_layout=True)
    plt.scatter(score1['month'], score1['norm_repincd'], label="Reference",color='k')
    plt.plot(score1['month'],score1['norm_simincd'],label="Simulation")
    plt.legend()
    plt.xlabel("Month")
    plt.ylabel("Normalized Clinical Incidence")
    #plt.ylim(0, 0.2)
    
    plt.savefig(os.path.join(plt_dir,f"incidence_{site}.png"))
    
    
def save_maxEIR(site="", wdir="./"):
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"InsetChart.csv"))
    sim_df = sim_df.rename(columns={'Sample_ID':'param_set'})
  
    best = pd.read_csv(f"{wdir}/emod.best.csv")
    best = best['param_set'][0]
    sim_df = sim_df[sim_df['param_set']==best]

    #print(sim_df)
    sim_df['year'] = [np.trunc(t/365) for t in sim_df['time']]
    sim_df['month'] = sim_df.apply(lambda row: np.trunc((row['time'] - (row['year']*365))/30.001)+1, axis=1)
    last_year= max(sim_df['year'])
    sim_df = sim_df[sim_df['year'] >= last_year-10]
    sim_df = sim_df.groupby(['year','month','Run_Number','param_set']).agg(monthEIR=('Daily EIR', 'sum')).reset_index()
    sim_df = sim_df.groupby('param_set').agg(maxEIR=('monthEIR','max')).reset_index()
    return sim_df

def save_AnnualIncidence(site="", wdir="./",agebin=5):
    ### Load analyzed monthly MalariaSummaryReport from simulation
    sim_cases = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"PfPR_ClinicalIncidence_monthly.csv"))
    #print(sim_cases)
    sim_cases['Inc'] = sim_cases['Cases'] / sim_cases['Pop']
    sim_cases = sim_cases.groupby(['Sample_ID', 'Year','agebin'])['Inc'].agg(np.nanmean).reset_index()
    # filter to age 5+
    sim_cases = sim_cases[sim_cases['agebin']==agebin]
    ##### Score - Mean annual cases per person #####
    ################################################
    # get average annual incidence by year (across months) and then across years
    print(sim_cases)
    score2 = sim_cases.groupby(['Sample_ID','agebin','Year'])['Inc'].agg(np.nansum).reset_index()
    score2 = score2.groupby(['Sample_ID','agebin'])['Inc'].agg(np.nanmean).reset_index()
    sim_cases = sim_cases.rename(columns={'Sample_ID':'param_set'})
    best = pd.read_csv(f"{wdir}/emod.best.csv")
    best = best['param_set'][0]
    aci = sim_cases[sim_cases['param_set']==best]
    
    return aci

def plot_prevalence(site="",plt_dir=os.path.join(manifest.simulation_output_filepath,"_plots"),wdir='./',start_year=0):
  
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"InsetChart.csv"))
    sim_df = sim_df.rename(columns={'Sample_ID':'param_set'})
  
    best = pd.read_csv(f"{wdir}/emod.best.csv")
    best = best['param_set'][0]
    sim_df = sim_df[sim_df['param_set']==best]
    sim_df['date'] = [timedelta(days=t) + datetime.strptime("19600101", '%Y%m%d') for t in sim_df['time']]
    sim_df2 = sim_df

    #print(sim_df)
    sim_df['year'] = [np.trunc(t/365) for t in sim_df['time']]
    sim_df['month'] = sim_df.apply(lambda row: np.trunc((row['time'] - (row['year']*365))/30.001)+1, axis=1)

    sim_df['year'] = sim_df['year'] + start_year 
    sim_df = sim_df[sim_df['month']<=12]
    
    refpcr = pd.read_csv(os.path.join(manifest.base_reference_filepath,
                                      coord_df.at['prevalence_comparison_reference','value']))
    ref_date_format = '%m/%d/%Y'
    refpcr['month'] = np.nan
    refpcr['day'] = np.nan
    refpcr['year'] = np.nan
    for index, row in refpcr.iterrows():
        #print(row['date'])
        dayof = datetime.strptime(row['date'], ref_date_format)
        refpcr['date'][index]= dayof
        #print(dayof)
        #print(dayof.month)
        refpcr['month'][index] = dayof.month
        #print(dayof.month)
        refpcr['day'][index] = dayof.day
        #print(dayof.month)
        refpcr['year'][index] = dayof.year
    # print(sim_df)
    # print(refpcr)
    #exit(1)
    plt.figure(1, figsize=(6, 6), dpi=300, tight_layout=True)
    plt.scatter(refpcr['date'], refpcr['ref_prevalence'], label="Reference")
    plt.plot(sim_df['date'],sim_df['PCR Parasite Prevalence'], label="Simulation")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("PCR Parasite Prevalence")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(plt_dir,f"prevalence_{site}.png"))



if __name__ == "__main__":

    workdir="/projects/b1139/environmental_calibration/simulations/output/241007_test2/"
    site="Nanoro"
    n=0
    #print(compute_scores_across_site(site))
    # mEIR = save_maxEIR(site=site, wdir = f"{workdir}/LF_{n}")
    # mEIR.to_csv(f"{workdir}/LF_{n}/maxEIR.csv")
    # ACI = save_AnnualIncidence(site=site, wdir =f"{workdir}/LF_{n}")
    # ACI.to_csv(f"{workdir}/LF_{n}/ACI.csv")
    #plot_incidence(site=site, plt_dir=os.path.join(f"{workdir}/LF_{n}"), wdir=os.path.join(f"{workdir}/LF_{n}"),agebin=5)
    plot_prevalence(site=site, plt_dir=os.path.join(f"{workdir}/LF_{n}"), wdir=os.path.join(f"{workdir}/LF_{n}"))

