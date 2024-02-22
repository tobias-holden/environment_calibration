import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import os
import manifest as manifest
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from datetime import timedelta
sns.set_context("talk")
sns.set_style("white")

from compare_to_data.calculate_all_scores import  compute_all_scores, load_case_data

def compute_scores_across_site(site):
    
    scores = compute_all_scores(site)
    print("Replacing NA")
    scores = scores.fillna(1.0)
    
    print("Applying weights...")
    weights = pd.read_csv(os.path.join(manifest.input_files_path,"my_weights.csv"),index_col=0)
    print(weights)
    scores['shape_score'] = [float(weights.at['shape_score','weight'])*val for val in scores['shape_score']]
    scores['eir_score'] = [float(weights.at['eir_score','weight'])*val for val in scores['eir_score']]
    scores['intensity_score'] = [float(weights.at['intensity_score','weight'])*val for val in scores['intensity_score']]
    scores['prevalence_score'] = [float(weights.at['prevalence_score','weight'])*val for val in scores['prevalence_score']]
    
    scores = scores.rename(columns={"Sample_ID":"param_set"})
    print("Weighted Scores")
    print(scores)
    return scores

def plot_incidence(round_num=0,site="",plt_dir=os.path.join(manifest.simulation_output_filepath,"_plots"),wdir='.'):
    
    best = pd.read_csv(os.path.join(wdir,"emod.best.csv"))
    best = best['param_set'][0]
    
    #sim_cases = pd.read_csv(os.path.join(manifest.simulation_output_filepath,f"CopyOf{site}","PfPR_ClinicalIncidence_monthly.csv"))
    sim_cases = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"PfPR_ClinicalIncidence_monthly.csv"))
    sim_cases = sim_cases[sim_cases['Sample_ID']==best]
    sim_cases = sim_cases[sim_cases['agebin']==5]
    sim_cases = sim_cases.groupby(['Sample_ID', 'Year', 'month'])[['Pop','Cases']].agg(np.nanmean).reset_index()
    sim_cases = sim_cases.rename(columns={'Year':'year'})
    
    #sim_treated = pd.read_csv(os.path.join(manifest.simulation_output_filepath,f"CopyOf{site}","events.csv"))
    sim_treated = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site, "events.csv"))
    sim_treated = sim_treated[sim_treated['Event_Name']=="Received_Treatment"]
    sim_treated = sim_treated[sim_treated['Sample_ID']==best]
    sim_treated['Age'] = [np.trunc(a/365) for a in sim_treated['Age']]
    sim_treated = sim_treated[sim_treated['Age'] >= 5]
    sim_treated['year'] = [np.trunc(t/365) for t in sim_treated['Time']]
    sim_treated['month'] = sim_treated.apply(lambda row: (row['Time'] - row['year']*365), axis=1)
    sim_treated['month'] = [np.trunc(m/30.01)+1 for m in sim_treated['month']]
    sim_treated['month']= sim_treated.apply(lambda row: np.trunc((row['Time'] - (row['year']*365))/30.001)+1, axis=1)
    sim_treated=sim_treated[sim_treated['month']<=12]
    sim_treated=sim_treated.groupby(['Sample_ID', 'year','month','Run_Number']).agg("size").reset_index(name='counts')
    sim_treated['year'] = sim_treated['year']#+start_year
    sim_treated1 = sim_treated.merge(sim_cases, on=['Sample_ID', 'year','month'], how='inner')
    sim_treated1['simincd'] = sim_treated1['counts'] / sim_treated1['Pop']
    sim_treated1 = sim_treated1.groupby(['Sample_ID','month'])['simincd'].agg(np.nanmean).reset_index(name='simincd')
    #it would be useful to make sure that this is dropping NA in norm_simincd and not just sampleid/month
    sim_cases1 = sim_treated1
    sim_cases1=sim_cases1.merge(sim_cases1.groupby(['Sample_ID'])['simincd'].agg(np.nanmax).reset_index(name='max_simincd'), on=['Sample_ID'],how='left').reset_index()
    sim_cases1['norm_simincd'] = sim_cases1.apply(lambda row: (row['simincd'] / row['max_simincd']), axis=1)
    sim_cases1 = sim_cases1.groupby(['Sample_ID', 'month'])['norm_simincd'].agg(np.nanmean).reset_index()
    
    
    
    rcases = load_case_data()
    # will need to update the simulation coordinator so sapone isnt hardcoded
    rcases = rcases[rcases['DS_Name']==site]
    rcases = rcases[rcases['age']=='ov5']
    #rcases['year'] = [(y - start_year) for y in rcases['year']]
    rcases=rcases.merge(rcases.groupby('year')['repincd'].agg(np.nanmax).reset_index(name='max_incd'), on='year',how='left')
    
    rcases['norm_repincd'] = rcases.apply(lambda row: (row['repincd'] / row['max_incd']), axis=1)
    
    rcases1 = rcases.groupby('month')['norm_repincd'].agg(np.nanmean).reset_index()
    plt.figure(3, figsize=(6, 6), dpi=300, tight_layout=True)
    plt.plot(sim_cases1['month'],sim_cases1['norm_simincd'],label=f"{round_num}-{best}")
    if round_num ==0 :
        plt.scatter(rcases1['month'], rcases1['norm_repincd'], label="Reference",color='k')
    plt.legend()
    plt.xlabel("Month")
    plt.ylabel("Normalized Clinical Incidence")
    plt.ylim(0, 1.1)
    plt.xticks(ticks=[1,2,3,4,5,6,7,8,9,10,11,12])
    plt.savefig(os.path.join(plt_dir,f"incidence_{site}.png"))
    plt.savefig(os.path.join(plt_dir,f"incidence_{site}.pdf"))
    plt.close(3)
    
    plt.figure(4, figsize=(6, 6), dpi=300, tight_layout=True)
    plt.plot(sim_cases1['month'],sim_cases1['norm_simincd'],label=f"{round_num}-{best}")
    if round_num ==0 :
        plt.scatter(rcases1['month'], rcases1['norm_repincd'], label="Reference",color='k')
    else:
        plt.scatter(rcases1['month'], rcases1['norm_repincd'],color='k')

    plt.legend()
    plt.xlabel("Month")
    plt.ylabel("Normalized Clinical Incidence")
    plt.ylim(0, 1.1)
    plt.xticks(ticks=[1,2,3,4,5,6,7,8,9,10,11,12])
    plt.savefig(os.path.join(plt_dir,f"incidence_{site}_all.png"))
    plt.savefig(os.path.join(plt_dir,f"incidence_{site}_all.pdf"))
    
    
    
    
def save_EIR(site="", wdir="./"):
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"InsetChart.csv"))
    sim_df = sim_df.rename(columns={'Sample_ID':'param_set'})
    best = pd.read_csv(f"{wdir}/emod.best.csv")
    best = best['param_set'][0]
    sim_df = sim_df[sim_df['param_set']==best]
    sim_df['year'] = [np.trunc(t/365) for t in sim_df['time']]
    sim_df['month'] = sim_df.apply(lambda row: np.trunc((row['time'] - (row['year']*365))/30.001)+1, axis=1)
    last_year= max(sim_df['year'])
    sim_df = sim_df[sim_df['year'] >= last_year-10]
    # Sum daily to monthly EIR
    sim_df = sim_df.groupby(['month','year','Run_Number','param_set']).agg(EIR=('Daily EIR', 'sum')).reset_index()
    # Average across Run Numbers
    sim_df = sim_df.groupby(['param_set','year','month']).agg(EIR=('EIR','mean')).reset_index()
    # Find min/max for parameter set
    sim_df = sim_df.groupby('param_set').agg({'EIR': ['min', 'max']}).droplevel(axis=1, level=0)
    sim_df = sim_df.reset_index()
    
    return sim_df

def save_AnnualIncidence(site="", wdir="./"):
    ### Load analyzed monthly MalariaSummaryReport from simulation
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"PfPR_ClinicalIncidence_monthly.csv"))
    sim_df = sim_df.rename(columns={'Sample_ID':'param_set'})
    best = pd.read_csv(f"{wdir}/emod.best.csv")
    best = best['param_set'][0]
    sim_df = sim_df[sim_df['param_set']==best]
    # filter to age 5+
    sim_df = sim_df[sim_df['agebin']==5]
    # get mean population and clinical cases by month, year, and param_set
    sim_df = sim_df.groupby(['param_set', 'Year', 'month'])[['Pop','Cases']].agg(np.nanmean).reset_index()
    sim_df = sim_df.rename(columns={'Year':'year'}) # lower case
    
    aci = sim_df.groupby(['param_set', 'year'])['Cases'].agg(np.nanmean).reset_index()
    aci = sim_df.groupby(['param_set'])['Cases'].agg(np.nanmean).reset_index()
    # calculate incidence row-wise
    return aci

def plot_prevalence(round_num=0, site="",plt_dir=os.path.join(manifest.simulation_output_filepath,"_plots"),wdir='./'):
  
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath,site,"InsetChart.csv"))
    sim_df = sim_df.rename(columns={'Sample_ID':'param_set'})
  
    best = pd.read_csv(f"{wdir}/emod.best.csv")
    best = best['param_set'][0]
    sim_df = sim_df[sim_df['param_set']==best]
    sim_df['date'] = [timedelta(days=t) + datetime.strptime("19600101", '%Y%m%d') for t in sim_df['time']]
    #print(sim_df)
    sim_df['month'] = np.nan
    sim_df['day'] = np.nan
    sim_df['year'] = np.nan
    for index, row in sim_df.iterrows():
        #print(row['date'])
        dayof = sim_df['date'][index]
        sim_df['month'][index] = dayof.month
        #print(dayof.month)
        sim_df['day'][index] = dayof.day
        #print(dayof.month)
        sim_df['year'][index] = dayof.year
    
    #sim_df['year'] = [np.trunc(t/365) for t in sim_df['time']]
    #sim_df['month'] = sim_df.apply(lambda row: np.trunc((row['time'] - (row['year']*365))/30.001)+1, axis=1)

    #sim_df['year'] = sim_df['year'] #+ start_year 
    #sim_df = sim_df[sim_df['month']<=12]
    
    refpcr = pd.read_csv(os.path.join(manifest.PROJECT_DIR,"reference_datasets","Nanoro_microscopy_prevalence_U5.csv"))
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

    #print(sim_df)
    sim_df = sim_df[sim_df['year']>=(max(sim_df['year'])-5)]
    #print(sim_df)
    sim_df = sim_df.groupby('date')['PCR Parasite Prevalence'].agg('mean').reset_index(name='prevalence') 
    #print(sim_df)
    plt.figure(1, figsize=(6, 6), dpi=300, tight_layout=True)
    
    plt.plot(sim_df['date'], sim_df['prevalence'],label=f"{round_num}-{best}")
    if round_num ==0 :
        plt.scatter(refpcr['date'], refpcr['ref_prevalence'], label="Reference", c='k')
    else: 
        plt.scatter(refpcr['date'], refpcr['ref_prevalence'], c='k')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize="10")
    plt.xlabel("Date")
    plt.ylabel("PCR Parasite Prevalence")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')

    plt.savefig(os.path.join(plt_dir,f"prevalence_monthly_{site}.png"),bbox_inches='tight')
    plt.savefig(os.path.join(plt_dir,f"prevalence_monthly_{site}.pdf"),bbox_inches='tight')
    plt.close(1)
    
    plt.figure(2, figsize=(6, 6), dpi=300, tight_layout=True)
    
    plt.plot(sim_df['date'], sim_df['prevalence'],label=f"{round_num}-{best}")
    if round_num ==0 :
        plt.scatter(refpcr['date'], refpcr['ref_prevalence'], label="Reference", c='k')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize="10")
    plt.xlabel("Date")
    plt.ylabel("PCR Parasite Prevalence")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(plt_dir,f"prevalence_monthly_{site}_all.png"),bbox_inches='tight')
    plt.savefig(os.path.join(plt_dir,f"prevalence_monthly_{site}_all.pdf"),bbox_inches='tight')
    
    
# 
#     if param_sets_to_plot is None:
#         param_sets_to_plot = list(set(scored["param_set"]))
# 
#     #todo Add error bars on data
#     # combined_df["reference_std"] = np.sqrt(combined_df["reference"])
#     scored["reference_std"] = np.sqrt(combined_df["ref_pop_size"]*combined_df["reference"])/combined_df["ref_pop_size"]
# 
#     plt.figure()
#     plt.plot(combined_df["mean_age"], combined_df["reference"], label="reference", marker='o')
#     for param_set, sdf in combined_df.groupby("param_set"):
#         if param_set in param_sets_to_plot:
#             plt.plot(sdf["mean_age"], sdf["simulation"], label=f"Param set {param_set}", marker='s')
#     plt.xlabel("Age")
#     plt.ylabel("Incidence per person per year")
#     plt.title(site)
#     plt.legend()
#     plt.tight_layout()
#     #plt.savefig(os.path.join(manifest.simulation_output_filepath, "_plots", f"incidence_{site}.png"))
#     plt.savefig(os.path.join(plt_dir,f"incidence_{site}.png"))


def plot_all_comparisons(param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    #plot_incidence_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)
    #plot_prevalence(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)
    #plot_density_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)
    # plot_infectiousness_comparison_all_sites(param_sets_to_plot=param_sets_to_plot) #fixme Not implemented yet
    return 
   
if __name__ == "__main__":
    #plot_all_comparisons(param_sets_to_plot=[1.0,4.0,6.0,9.0,15.0])
    
    #if you are running directly from run_full_comparison you are going to probably want to 
    #manually add a different default numOf_param_sets, for example, numOf_param_sets = 16
    workdir="/projects/b1139/environmental_calibration/simulations/output/Nanoro_test1"
    n=0
    #print(compute_scores_across_site("Sapone"))
    # mEIR = save_maxEIR(site="Sapone", wdir = f"{workdir}/LF_{n}")
    # mEIR.to_csv(f"{workdir}/LF_{n}/maxEIR.csv")
    # ACI = save_AnnualIncidence(site="Sapone", wdir =f"{workdir}/LF_{n}")
    # ACI.to_csv(f"{workdir}/LF_{n}/ACI.csv")
    plot_prevalence(site="Nanoro", plt_dir=os.path.join(f"{workdir}/LF_{n}"), wdir=os.path.join(f"{workdir}/LF_{n}"))
    plot_incidence(site="Nanoro", plt_dir=os.path.join(f"{workdir}/LF_{n}"), wdir=os.path.join(f"{workdir}/LF_{n}"))
