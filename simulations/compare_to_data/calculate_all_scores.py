from math import sqrt
import os
from statistics import mean
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from simulations import manifest

from simulations.load_inputs import load_sites
from simulations.helpers import load_coordinator_df
from datetime import datetime

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


def compute_all_scores(site):
    ### Load analyzed monthly MalariaSummaryReport from simulation
    sim_cases = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"PfPR_ClinicalIncidence_monthly.csv"))
    #print(sim_cases)
    # filter to age 5+
    sim_cases = sim_cases[sim_cases['agebin']==5]
    # get mean population and clinical cases by month, year, and Sample_ID
    sim_cases = sim_cases.groupby(['Sample_ID', 'Year', 'month'])[['Pop','Cases']].agg(np.nanmean).reset_index()
    sim_cases = sim_cases.rename(columns={'Year':'year'}) # lower case

    ### Load analyzed EventRecorder from simulation
    sim_treated = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site, "events.csv"))
    #print(sim_treated)
    # filter to "received_treatment" events
    sim_treated = sim_treated[sim_treated['Event_Name']=="Received_Treatment"]
    # convert age to years
    sim_treated['Age'] = [np.trunc(a/365) for a in sim_treated['Age']]
    # filter to age >=5
    sim_treated = sim_treated[sim_treated['Age'] >= 5]
    # convert time to year, and month
    sim_treated['year'] = [np.trunc(t/365) for t in sim_treated['Time']]
    sim_treated['month']= sim_treated.apply(lambda row: np.trunc((row['Time'] - (row['year']*365))/30.001)+1, axis=1)
    sim_treated=sim_treated[sim_treated['month']<=12]
    #print(sim_treated)
    # count # of events by month/year/run_number/sample_id
    sim_treated=sim_treated.groupby(['Sample_ID', 'year','month','Run_Number']).agg("size").reset_index(name='counts')
    #print(sim_cases)
    #print(sim_treated)
    # merge sim_treated with sim_cases for population
    sim_treated1 = sim_treated.merge(sim_cases, on=['Sample_ID', 'year','month'], how='inner')
    #print(sim_treated1)
    # calculate incidence
    sim_treated1['simincd'] = sim_treated1['counts'] / sim_treated1['Pop']
    #print(sim_treated1)
    # get average incidence by sample_ID/month (across run_numbers & years)
    sim_treated1 = sim_treated1.groupby(['Sample_ID','month'])['simincd'].agg(np.nanmean).reset_index(name='simincd')
    #print(sim_treated1)
    sim_cases1 = sim_treated1
    #print(sim_cases1)
    # calculate max of simincd per sample_ID
    sim_cases1=sim_cases1.merge(sim_cases1.groupby(['Sample_ID'])['simincd'].agg(np.nanmax).reset_index(name='max_simincd'), on=['Sample_ID'],how='left').reset_index()
    #print(sim_cases1)
    # normalized against maximum
    sim_cases1['norm_simincd'] = sim_cases1.apply(lambda row: (row['simincd'] / row['max_simincd']), axis=1)
    #print(sim_cases1)
    # get mean normalized incidence by month/sample_id (across years)
    sim_cases1 = sim_cases1.groupby(['Sample_ID', 'month'])['norm_simincd'].agg(np.nanmean).reset_index()
    #print(sim_cases1)
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

    ### Load InsetChart.csv
    ic = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site, "InsetChart.csv"))
    # convert Time to year and month
    ic['year'] = [np.trunc(t/365) for t in ic['time']]
    ic['month'] = ic.apply(lambda row: np.trunc((row['time'] - (row['year']*365))/30.001)+1, axis=1)
    ic = ic[ic['month']<=12]
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
    
    #### SCORE - Monthly PCR Parasite Prevalence ####
    #################################################
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

    #### Load incidence data
    rcases = load_case_data()
    # filter to DS_Name
    rcases = rcases[rcases['DS_Name']==site]
    # filter to age ov5
    rcases = rcases[rcases['age']=='ov5']
    # convert rcases 'year' to start at 0, like simulations
    rcases['year'] = [(y - start_year) for y in rcases['year']]
    # sum incidence across year    
    rcases=rcases.merge(rcases.groupby('year')['repincd'].agg(np.nanmax).reset_index(name='max_incd'), on='year',how='left')
    # normalize monthly incidence so each year sums to 1
    rcases['norm_repincd'] = rcases.apply(lambda row: (row['repincd'] / row['max_incd']), axis=1)
    # get average normalized incidence per month
    rcases1 = rcases.groupby('month')['norm_repincd'].agg(np.nanmean).reset_index()
    #print(rcases1)
    ##### Score - Monthly Incidence Shape #####
    ###########################################
    #print(sim_cases1)
    # merge simulated normalized monthly incidence with reference data on ['month']
    score1 = sim_cases1.merge(rcases1, on ='month')
    # score as (difference)^2
    #print(score1)
    score1['shape_score'] = score1.apply(lambda row: ((row['norm_simincd'] - row['norm_repincd'])**2), axis=1)
    # get average score by sample_id
    #print(score1)
    score1=score1.groupby('Sample_ID')['shape_score'].agg(np.nanmean).reset_index()
    #print(score1)
    ##### Score - Mean annual cases per person #####
    ################################################
    # get average annual incidence by year (across months) and then across years
    score2 = sim_cases.groupby(['Sample_ID','year'])['Cases'].agg(np.nanmean).reset_index()
    score2 = sim_cases.groupby(['Sample_ID'])['Cases'].agg(np.nanmean).reset_index()
    # Compare to target of 4.3
    score2['intensity_score'] = score2.apply(lambda row: abs(row['Cases']-4.3)/4.3, axis=1)
    
   
    # merge unweighted scores into one dataframe, and return
    scores = score1.merge(score2[['Sample_ID','intensity_score']], how='outer', on='Sample_ID')
    scores = scores.merge(score3[['Sample_ID','prevalence_score']], how='outer', on='Sample_ID')
    scores = scores.merge(score4[['Sample_ID','eir_score']], how='outer', on='Sample_ID')
    print("Raw Scores")
    print(scores)
    return scores
  
if __name__ == '__main__':
  calculate_all_scores("Sapone")

