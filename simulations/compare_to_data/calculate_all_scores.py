import sys
sys.path.append('../')
from math import sqrt, exp
import os
from statistics import mean
import numpy as np
import pandas as pd
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'

from simulations import manifest

from simulations.load_inputs import load_sites
from simulations.helpers import load_coordinator_df
from datetime import datetime

import math
from scipy.special import gammaln

coord_csv = load_coordinator_df(characteristic=False, set_index=True)

all_sites = []
sites, nSims = load_sites()
for site in sites:
    if coord_csv.at[site, 'calculate_scores'] == 1 :
        all_sites.append(site)

start_year = 1960

def load_case_data():
    #improvement, have this be a simulation coordinator feature, to grab the correct file with routine_data
    case_df= pd.read_csv(os.path.join(manifest.PROJECT_DIR,'reference_datasets','routine_seb_agg_confpres.csv'))
    case_df = case_df.rename(columns={'Date': 'date', 'case': 'repincd'})
    ##print(case_df)
    ref_format = '%Y-%m-%d'
    case_df['month'] = np.nan
    case_df['day'] = np.nan
    case_df['year'] = np.nan
    for index, row in case_df.iterrows():
        ##print(row['date'])
        dayof = datetime.strptime(row['date'], ref_format)
        case_df.loc[index,'month'] = dayof.month
        case_df.loc[index,'day'] = dayof.day
        case_df.loc[index,'year'] = dayof.year
        
    
    return(case_df)

def load_prevalence_data(site):
    ### Load reference PCR prevalence data
    refpcr = pd.read_csv(os.path.join(manifest.PROJECT_DIR,"reference_datasets","Nanoro_microscopy_prevalence_U5.csv"))
    # convert date to month, day, and year
    ref_date_format = '%m/%d/%Y'
    refpcr['month'] = np.nan
    refpcr['day'] = np.nan
    refpcr['year'] = np.nan
    for index, row in refpcr.iterrows():
        dayof = datetime.strptime(row['date'], ref_date_format)
        refpcr['month'][index] = dayof.month
        refpcr['day'][index] = dayof.day
        refpcr['year'][index] = dayof.year
    return(refpcr)

def prepare_inset_chart_data(site):
    ### Load InsetChart.csv
    ic = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site, "InsetChart.csv"))
    # convert Time to year and month
    ic['year'] = [np.trunc(t/365) for t in ic['time']]
    ic['month'] = ic.apply(lambda row: np.trunc((row['time'] - (row['year']*365))/30.001)+1, axis=1)
    ic = ic[ic['month']<=12]
    return(ic)
  
def compare_all_age_PCR_prevalence(site):
    #### SCORE - Monthly PCR Parasite Prevalence ####
    #################################################
    ic = prepare_inset_chart_data(site)
    refpcr = load_prevalence_data(site)
    # get average prevalence from insetchart for each year/month/sample_id 
    score3 = ic.groupby(['Sample_ID','year','month']).agg(prevalence=('PCR Parasite Prevalence', 'mean')).reset_index()
    # convert reference_pcr 'year' to start at 0, like simulations
    refpcr['year'] = [(y - start_year) for y in refpcr['year']]
    # merge with reference data on ['year','month']
    score3 = score3[['year', 'month', 'Sample_ID', 'prevalence']]
    score3 = score3.merge(refpcr,on=["year","month"], how='left')
    # score as sqrt(|diff|)
    score3['prevalence_score'] = score3.apply(lambda row: sqrt(abs(row['prevalence']-row['ref_prevalence'])**2), axis=1)
    # get average score by sample_id
    score3 = score3.groupby('Sample_ID').agg(prevalence_score=('prevalence_score','mean')).reset_index()
    return score3

def check_EIR_threshold(site):
    ic = prepare_inset_chart_data(site)
    # get maximum year of simulation
    last_year= max(ic['year'])
    
    #### SCORE - Maximum monthly EIR threshold ####
    ###############################################
    # filter insetchart to last 10 years of simulation
    score4 = ic[ic['year'] >= last_year-10]
    # sum daily to monthly EIR
    score4 = score4.groupby(['Sample_ID','Run_Number','month','year']).agg(EIR=('Daily EIR','sum')).reset_index()
    # Average across Run Numbers
    score4 = score4.groupby(['Sample_ID','year','month']).agg(EIR=('EIR','mean')).reset_index()
    # Find min/max for parameter set
    score4 = score4.groupby('Sample_ID').agg({'EIR': ['min', 'max']}).droplevel(axis=1, level=0).reset_index()
    score4['eir_score'] = 1*((score4['max']>=100) | (score4['min']==0))
    #print(score4.to_string())
    #print(score4)
    return score4

def compute_incidence_likelihood(combined_df):
    df = combined_df
    df['ref.Trials'] = 1000
    df['sim.Trials'] = 1000
    df['ref.Observations'] = df['ref.Trials'] * df['norm_repincd']
    df['sim.Observations'] = df['sim.Trials'] * df['norm_simincd']
    df['ll'] = gammaln(df['ref.Trials'] + 1) + gammaln(df['sim.Trials'] + 2) - gammaln(df['ref.Trials'] + df['sim.Trials'] + 2) + gammaln(
        df['ref.Observations'] + df['sim.Observations'] + 1) + gammaln(
        df['ref.Trials'] - df['ref.Observations'] + df['sim.Trials'] - df['sim.Observations'] + 1) - gammaln(df['ref.Observations'] + 1) - gammaln(
        df['ref.Trials'] - df['ref.Observations'] + 1) - gammaln(df['sim.Observations'] + 1) - gammaln(df['sim.Trials'] - df['sim.Observations'] + 1)
    #print(ll)
    return df #mean

def compare_incidence_shape(site,agebin):
    #### Load incidence data
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

    ### Load analyzed EventRecorder from simulation
    # print(score1)
    # exit(1)
    # sim_treated = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site, "events.csv"))
    # sim_treated = sim_treated[sim_treated['Event_Name']=="Received_Treatment"]
    # # convert age to years
    # sim_treated['Age'] = [np.trunc(a/365) for a in sim_treated['Age']]
    # # filter to age >=5
    # sim_treated = sim_treated[sim_treated['Age'] >= 5]
    # # convert time to year, and month
    # sim_treated['year'] = [np.trunc(t/365) for t in sim_treated['Time']]
    # sim_treated['month']= sim_treated.apply(lambda row: np.trunc((row['Time'] - (row['year']*365))/30.001)+1, axis=1)
    # sim_treated=sim_treated[sim_treated['month']<=12]
    # # count # of events by month/year/run_number/sample_id
    # sim_treated=sim_treated.groupby(['Sample_ID', 'year','month','Run_Number']).agg("size").reset_index(name='counts')
    # # merge sim_treated with sim_cases for population
    # sim_treated1 = sim_treated.merge(sim_cases, on=['Sample_ID', 'year','month'], how='inner')
    # # calculate incidence
    # sim_treated1['simincd'] = sim_treated1['counts'] / sim_treated1['Pop']
    # # get average incidence by sample_ID/month (across run_numbers & years)
    # sim_treated1 = sim_treated1.groupby(['Sample_ID','month'])['simincd'].agg(np.nanmean).reset_index(name='simincd')
    # sim_cases1 = sim_treated1
    # # calculate max of simincd per sample_ID
    # sim_cases1=sim_cases1.merge(sim_cases1.groupby(['Sample_ID'])['simincd'].agg(np.nanmax).reset_index(name='max_simincd'), on=['Sample_ID'],how='left').reset_index()
    # # normalized against maximum
    # sim_cases1['norm_simincd'] = sim_cases1.apply(lambda row: (row['simincd'] / row['max_simincd']), axis=1)
    # # get mean normalized incidence by month/sample_id (across years)
    # sim_cases1 = sim_cases1.groupby(['Sample_ID', 'month'])['norm_simincd'].agg(np.nanmean).reset_index()
    # # merge simulated normalized monthly incidence with reference data on ['month']
    # score1 = sim_cases1.merge(rcases, on ='month')
    # score as log-likelihood
    ll = compute_incidence_likelihood(score1)
    ll = ll.groupby(['Sample_ID'])['ll'].agg(np.nansum).reset_index()
    score1 = score1.drop(columns=['ll']).merge(ll,on='Sample_ID')
    #print(score1)
    score1=score1.groupby(['Sample_ID'])['ll'].agg(np.nanmean).reset_index()
    score1.rename(columns={'ll': 'shape_score'}, inplace=True)
    score1['shape_score'] = score1['shape_score'].abs()
    #print(score1)
    #print(score1)
    return score1
  
def compare_annual_incidence(site,agebin):
    rcases = load_case_data()
    rcases = rcases[rcases['DS_Name']==site]
    rcases['cases']=rcases['repincd'] / 10000
    rcases = rcases.groupby(['age','year'])[['cases']].agg(np.nansum).reset_index()
    target=rcases[rcases['age']=='ov5']
    target=target['cases'].mean()
    ### Load analyzed monthly MalariaSummaryReport from simulation
    sim_cases = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"PfPR_ClinicalIncidence_annual.csv"))
    #print(sim_cases)
    sim_cases['Inc'] = sim_cases['Cases'] / sim_cases['Pop']
    sim_cases = sim_cases.groupby(['Sample_ID', 'Year','agebin'])['Inc'].agg(np.nanmean).reset_index()
    # filter to age 5+
    sim_cases = sim_cases[sim_cases['agebin']==agebin]
    ##### Score - Mean annual cases per person #####
    ################################################
    # get average annual incidence by year (across months) and then across years
    #print(sim_cases)
    score2 = sim_cases.groupby(['Sample_ID','agebin'])['Inc'].agg(np.nanmean).reset_index()
    #print(score2)
    # Compare to target
    score2['intensity_score'] = score2.apply(lambda row: exp(abs(row['Inc']-target)/target), axis=1)
    #print(score2)
    return score2

def compute_all_scores(site):
    # merge unweighted scores into one dataframe, and return
    score1 = compare_incidence_shape(site,agebin=5)
    score2 = compare_annual_incidence(site,agebin=5)
    score3 = compare_all_age_PCR_prevalence(site)
    score4 = check_EIR_threshold(site)
    
    scores = score1.merge(score2[['Sample_ID','intensity_score']], how='outer', on='Sample_ID')
    scores = scores.merge(score3[['Sample_ID','prevalence_score']], how='outer', on='Sample_ID')
    scores = scores.merge(score4[['Sample_ID','eir_score']], how='outer', on='Sample_ID')
    #print(scores)
    return scores
  
  

if __name__ == '__main__':
  #compute_all_scores("Nanoro")
  compare_incidence_shape("Nanoro",agebin=5)
  print(compare_annual_incidence(site,agebin=5))
  print(compare_all_age_PCR_prevalence(site))
  #print(compare_annual_incidence(site="Nanoro",agebin=5))

