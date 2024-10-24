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
    
    scores['shape_score'] = scores['shape_score'].fillna(10)
    scores['intensity_score'] = scores['intensity_score'].fillna(10)
    scores['prevalence_score'] = scores['prevalence_score'].fillna(10)
    scores['eir_score'] = scores['eir_score'].fillna(10)
    
    scores = scores.rename(columns={"Sample_ID":"param_set"})
    #scores = pd.melt(scores, id_vars="param_set")
    #scores = scores.groupby("param_set")['value'].agg('sum').reset_index(name='score')
    
    return scores

def plot_incidence(site="",plt_dir=os.path.join(manifest.simulation_output_filepath,"_plots"),wdir='.',agebin=5):
    coord_df = load_coordinator_df()
    start_year = coord_df.at['simulation_start_year','value']
    best = pd.read_csv(os.path.join(wdir,"emod.best.csv"))
    best = best['param_set'][0]
    
    #### Load incidence data
    case_df = load_case_data()
    # filter to DS_Name
    case_df = case_df[case_df['site']==site]
    # filter to age of interest
    case_df = case_df[case_df['age']==agebin]
    # convert case_df 'year' to start at 0, like simulations
    case_df['year'] = [y - int(start_year) for y in case_df['year']]
    print(case_df)
    # sum incidence across year    
    case_df=case_df.merge(case_df.groupby('year')['case'].agg(np.nanmax).reset_index(name='max_incd'), on='year',how='left')
    # normalize monthly incidence so each year sums to 1
    case_df['norm_repincd'] = case_df.apply(lambda row: (row['case'] / row['max_incd']), axis=1)
    # get average normalized incidence per month
    rcases = case_df.groupby(['month','site'])['norm_repincd'].agg(np.nanmean).reset_index()
    
    sim_cases = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"ClinicalIncidence_monthly.csv"))
    # filter to age of interest
    sim_cases = sim_cases[sim_cases['agebin']==agebin]
    # get mean population and clinical cases by month, year, and Sample_ID
    sim_cases['Inc'] = sim_cases['Cases'] / sim_cases['Pop']
    sim_cases=sim_cases.merge(sim_cases.groupby(['Sample_ID','Year'])['Inc'].agg(np.nanmax).reset_index(name='max_simincd'), on=['Sample_ID'],how='left').reset_index()
    sim_cases['norm_simincd'] = sim_cases.apply(lambda row: (row['Inc'] / row['max_simincd']), axis=1)
    #print(sim_cases)
    # get mean normalized incidence by month/sample_id (across years)
    sim_cases = sim_cases.groupby(['Sample_ID', 'month'])['norm_simincd'].agg(np.nanmean).reset_index()
    # merge simulated normalized monthly incidence with reference data on ['month']
    case_df = sim_cases.merge(rcases, on ='month')
    case_df = case_df.dropna(subset=['norm_simincd']).reset_index()
    case_df = case_df[case_df['Sample_ID']==best]
    #print(rcases1)
    #print(sim_cases1)
    plt.clf()
    plt.figure(figsize=(6, 6), dpi=300, tight_layout=True)
    plt.plot(case_df['month'],case_df['norm_simincd'],label="Simulation")
    plt.scatter(case_df['month'], case_df['norm_repincd'], label="Reference",color='k')
    plt.legend()
    plt.xlabel("Month")
    plt.xticks(np.arange(1, 13))
    plt.ylabel("Normalized Clinical Incidence")
    plt.ylim(0, 1.1)
    plt.xlim()
    
    plt.savefig(os.path.join(plt_dir,f"incidence_{site}.png"))
    
    
def save_rangeEIR(site="", wdir="./"):
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
    sim_df = sim_df.groupby('param_set').agg(minEIR=('monthEIR','min'),maxEIR=('monthEIR','max')).reset_index()
    return sim_df

def save_AnnualIncidence(site="", wdir="./",agebin=5):
    ### Load analyzed monthly MalariaSummaryReport from simulation
    sim_cases = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"ClinicalIncidence_monthly.csv"))
    #print(sim_cases)
    sim_cases['Inc'] = sim_cases['Cases'] / sim_cases['Pop']
    sim_cases = sim_cases.groupby(['Sample_ID', 'Year','agebin'])['Inc'].agg(np.nanmean).reset_index()
    # filter to age
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

def plot_allAge_prevalence(site="",plt_dir=os.path.join(manifest.simulation_output_filepath,"_plots"),wdir='./'):
  
    coord_df=load_coordinator_df()
    start_year=int(coord_df.at['simulation_start_year','value'])
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"InsetChart.csv"))
    sim_df = sim_df.rename(columns={'Sample_ID':'param_set'})
  
    best = pd.read_csv(f"{wdir}/emod.best.csv")
    best = best['param_set'][0]
    sim_df = sim_df[sim_df['param_set']==best]
    sim_df['date'] = [timedelta(days=t) + datetime.strptime(f"{start_year}0101", '%Y%m%d') for t in sim_df['time']]
    sim_df2 = sim_df

    #print(sim_df)
    sim_df['year'] = [np.trunc(t/365) for t in sim_df['time']]
    sim_df['month'] = sim_df.apply(lambda row: np.trunc((row['time'] - (row['year']*365))/30.001)+1, axis=1)

    sim_df['year'] = sim_df['year'] + start_year 
    sim_df = sim_df[sim_df['month']<=12]
    
    refpcr = pd.read_csv(os.path.join(manifest.base_reference_filepath,
                                      coord_df.at['prevalence_comparison_reference','value']))
    # ref_date_format = '%m/%d/%Y'
    for index, row in refpcr.iterrows():
        m=refpcr['month'][index]
        d=refpcr['day'][index]
        y=refpcr['year'][index]
        refpcr['date'][index]= datetime.strptime(f"{y}{m}{d}", '%Y%m%d')
        # refpcr['month'][index] = dayof.month
        # refpcr['day'][index] = dayof.day
        # refpcr['year'][index] = dayof.year
    plt.clf()
    plt.figure(figsize=(6, 6), dpi=300, tight_layout=True)
    plt.scatter(refpcr['date'], refpcr['ref_prevalence'], label="Reference", color='k')
    plt.plot(sim_df['date'],sim_df['PCR Parasite Prevalence'], label="Simulation")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("PCR Parasite Prevalence")
    plt.ylim(0, 1)
    plt.gcf().autofmt_xdate()
    plt.savefig(os.path.join(plt_dir,f"prevalence_{site}.png"))


if __name__ == "__main__":

    workdir="/projects/b1139/environment_calibration/simulations/output/test_repo2/"
    site="Nanoro"
    n=0
    #print(compute_scores_across_site(site))
    #EIR = save_rangeEIR(site=site, wdir = f"{workdir}/LF_{n}")
    #EIR.to_csv(f"{workdir}/LF_{n}/EIR_range.csv")
    #ACI = save_AnnualIncidence(site=site, wdir =f"{workdir}/LF_{n}",agebin=100)
    #ACI.to_csv(f"{workdir}/LF_{n}/ACI.csv")
    ##plot_incidence(site=site, plt_dir=os.path.join(f"{workdir}/LF_{n}"), wdir=os.path.join(f"{workdir}/LF_{n}"),agebin=100)
    plot_allAge_prevalence(site=site, plt_dir=os.path.join(f"{workdir}/LF_{n}"), wdir=os.path.join(f"{workdir}/LF_{n}"))

