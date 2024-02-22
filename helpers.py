import os
import json
from functools import partial
import emod_api.config.default_from_schema_no_validation as dfs
import emod_api.demographics.Demographics as Demog
import emodpy_malaria.demographics.MalariaDemographics as Demographics
from emod_api.demographics.DemographicsTemplates import CrudeRate   
import emodpy_malaria.malaria_config as conf
import emodpy_malaria.malaria_config as malaria_config
from emodpy_malaria.interventions.treatment_seeking import add_treatment_seeking
from emodpy_malaria.interventions.drug_campaign import add_drug_campaign as dc
from emodpy_malaria.interventions.usage_dependent_bednet import add_scheduled_usage_dependent_bednet 
from emodpy_malaria.reporters.builtin import (
    add_event_recorder,
    add_malaria_summary_report,
    add_report_node_demographics,
    add_report_vector_stats
)
import pandas as pd
import manifest
import warnings
import numpy as np


def update_sim_random_seed(simulation, value):
    simulation.task.config.parameters.Run_Number = value
    return {"Run_Number": value}


def set_climate(config, shift):
    """
    Set climate to specific files, currently hardcoded.
    """
    config.parameters.Air_Temperature_Offset = shift
    config.parameters.Land_Temperature_Offset = shift
    return {"Temperature_Shift": shift}


def set_species_from_file(config, vector_file):
    """
    Set species and habitat level, assuming 1 species and 2 habitat type
    (1 input 1 constant)
    """
    # Multiple-Species Capability
    vdf = pd.read_csv(os.path.join(manifest.input_files_path,'vectors',vector_file))                                              
    vdf = vdf[vdf['node_id']== vdf['node_id'].unique()[0]]                                                                 
    s = [species for species in vdf['species']]                                                                            
    conf.add_species(config, manifest, s) 
    for r in range(len(s)):          
        conf.set_species_param(config, 
                               species = vdf['species'][r],
                               parameter='Anthropophily',
                               value=vdf['anthropophily'][r],
                               overwrite=True)
        conf.set_species_param(config, 
                               species = vdf['species'][r],
                               parameter='Indoor_Feeding_Fraction',
                               value=vdf['indoor_feeding'][r],
                               overwrite=True)                                                                                     
    return


def set_param_fn(config):
    config = conf.set_team_defaults(config, manifest)

    set_species_from_file(config, vector_file = "vectors.csv")
   
    
    config.parameters.Simulation_Duration = 70 * 365 + 1
    config.parameters.x_Temporary_Larval_Habitat = 1
    config.parameters.Age_Initialization_Distribution_Type = "DISTRIBUTION_COMPLEX"
    
    config.parameters.Air_Temperature_Filename = os.path.join(
        "climate", "air_temperature_daily.bin"
    )
    config.parameters.Land_Temperature_Filename = os.path.join(
        "climate", "air_temperature_daily.bin"
    )
    config.parameters.Rainfall_Filename = os.path.join(
        "climate", "rainfall_daily.bin"
    )
    config.parameters.Relative_Humidity_Filename = os.path.join(
        "climate", "relative_humidity_daily.bin"
    )
    config.parameters.Custom_Individual_Events = ["Received_Treatment","Received_SMC","Bednet_Using","Bednet_Discarded","Bednet_Got_New_One"]
    return config


def set_param(simulation, param, value):
    """
    Set specific parameter value
    Args:
        simulation: idmtools Simulation
        param: parameter
        value: new value
    Returns:
        dict
    """
    return simulation.task.set_parameter(param, value)


def add_outputs(task, site):
    """
    Requesting reports/outputs to the task.
    """
    coord_df = load_coordinator_df(characteristic=False, set_index=True)
    simulation_duration = coord_df.at[site,'simulation_duration']
    report_start = coord_df.at[site,'report_start_day']
    last_year = int(simulation_duration / 365)
    first_year = int(report_start / 365)
    sim_start_year = 0 #This may be useful to change? I don't know the input data
    for year in range(first_year, last_year):
        start_day = 0 + 365 * year
        sim_year = sim_start_year + year
        add_malaria_summary_report(
            task,
            manifest,
            start_day=start_day,
            end_day=365 + year * 365,
            reporting_interval=30,
            age_bins=[0.25, 5, 115],
            max_number_reports=13,
            pretty_format=True,
            filename_suffix=f"Monthly{sim_year}",
        )

    # add_vector_habitat_report(task, manifest)
    add_event_recorder(
        task,
        event_list=["Received_Treatment"],
        start_day=int(report_start),
        end_day=int(simulation_duration),
        min_age_years=0,
        max_age_years=50
    )

    # add_report_node_demographics(task, manifest,
    #                              age_bins = [0.25,5,50],
    #                              stratify_by_gender = False)
    # 
    # add_report_vector_stats(task, manifest,
    #                         species_list = ['arabiensis','funestus','gambiae'],
    #                         stratify_by_species = True,
    #                         include_death_state = False,
    #                         include_wolbachia = False,
    #                         include_gestation = False,
    #                         include_microsporidia = False)
    return


def add_hfca_hs(camp, hs_df):
    for r, row in hs_df.iterrows() :
        add_hs_from_file(camp, row)


def add_sapone_hs(camp,hs_df):
    for year in hs_df['year']:                                                                                     
        sub_df = hs_df[hs_df['year'] == year].reset_index()                                                      
        targets = [] 
        #print(year)                                                                                             
        for r in range(len(sub_df)) :                                                                            
            cm_coverage_by_age =  {'trigger': str(sub_df['trigger'][r]),                                         
                                   'coverage': float(sub_df['coverage'][r]),                                     
                                   'agemin': float(sub_df['age_min'][r]),                                        
                                   'agemax': float(sub_df['age_max'][r]),
                                   'rate': float(sub_df['rate'][r])}                                             
            targets.append(cm_coverage_by_age)                                                                   
        #print(targets)
        add_treatment_seeking(camp,                                                  
                              start_day = int(sub_df['start_day'][0]),                                        
                              duration = int(sub_df['duration'][0]),                                          
                              drug=['Artemether','Lumefantrine'],                                             
                              targets=targets,                                                                
                              broadcast_event_name="Received_Treatment")


def add_hs_from_file(camp, row):
    hs_child = row['U5_coverage']
    hs_adult = row['adult_coverage']
    severe_cases = row['severe_coverage']
    start_day = row['simday']
    duration = row['duration']
    if 'drug_code' in row.index:
        drug_code = row['drug_code']
    else:
        drug_code = 'AL'
    if drug_code == 'AL':
        drug = ['Artemether', 'Lumefantrine']
    elif drug_code == 'SP':
        drug = ['Sulfadoxine', 'Pyrimethamine']
    elif drug_code == 'CQ':
        drug = ['Chloroquine']
    elif drug_code == 'A':
        drug = ['Artemether']
    else:
        warnings.warn('Drug code not recognized. Assuming AL.')
        drug = ['Artemether', 'Lumefantrine']

    add_treatment_seeking(camp, start_day=start_day,
                          targets=[{'trigger': 'NewClinicalCase', 'coverage': hs_child, 'agemin': 0, 'agemax': 5,
                                   'rate': 0.3},
                                   {'trigger': 'NewClinicalCase', 'coverage': hs_adult, 'agemin': 5, 'agemax': 100,
                                    'rate': 0.3}],
                          drug=drug, duration=duration)
    add_treatment_seeking(camp, start_day=start_day,
                          targets=[{'trigger': 'NewSevereCase', 'coverage': severe_cases, 'rate': 0.5}], #change by adding column and reviewing literature
                          drug=drug, duration=duration)  # , broadcast_event_name='Received_Severe_Treatment')


def add_nmf_hs(camp, hs_df, nmf_df):
    # if no NMF rate is specified, assume all age groups have 0.0038 probability each day
    if nmf_df.empty:
        nmf_df = pd.DataFrame({'U5_nmf': [0.0038], 'adult_nmf': [0.0038]})
    elif nmf_df.shape[0] != 1:
        warnings.warn('The NMF dataframe has more than one row. Only values in the first row will be used.')
    nmf_row = nmf_df.iloc[0]

    # apply the health-seeking rate for clinical malaria to NMFs
    for r, row in hs_df.iterrows():
        add_nmf_hs_from_file(camp, row, nmf_row)


def add_nmf_hs_from_file(camp, row, nmf_row):
    hs_child = row['U5_coverage']
    hs_adult = row['adult_coverage']
    start_day = row['simday']
    duration = row['duration']
    if 'drug_code' in row.index:
        drug_code = row['drug_code']
    else:
        drug_code = 'AL'
    if start_day == 0:  # due to dtk diagnosis/treatment configuration, a start day of 0 is not supported
        start_day = 1  # start looking for NMFs on day 1 (not day 0) of simulation
        if duration > 1:
            duration = duration - 1
    nmf_child = nmf_row['U5_nmf']
    nmf_adult = nmf_row['adult_nmf']

    # workaround for maximum duration of 1000 days is to loop, creating a new campaign every 1000 days
    separate_durations = [1000] * int(np.floor(duration/1000))  # create a separate campaign for each 1000 day period
    if (duration - np.floor(duration/1000) > 0):  # add final remaining non-1000-day duration
        separate_durations = separate_durations + [int(duration - np.floor(duration/1000) * 1000)]
    separate_start_days = start_day + np.array([0] + list(np.cumsum(separate_durations)))
    for dd in range(len(separate_durations)):
        if nmf_child * hs_child > 0:
            dc(camp, 'MSAT', drug_code=drug_code, start_days=[separate_start_days[dd]],
                              target_group={'agemin': 0, 'agemax': 5},
                              coverage=nmf_child * hs_child,
                              repetitions=separate_durations[dd], tsteps_btwn_repetitions=1,
                              diagnostic_type='PF_HRP2', diagnostic_threshold=5,
                              receiving_drugs_event_name='Received_NMF_Treatment')
        if nmf_adult * hs_adult > 0:
            dc(camp, 'MSAT', drug_code=drug_code, start_days=[separate_start_days[dd]],
                              target_group={'agemin': 5, 'agemax': 120},
                              coverage=nmf_adult * hs_adult,
                              repetitions=separate_durations[dd], tsteps_btwn_repetitions=1,
                              diagnostic_type='PF_HRP2', diagnostic_threshold=5,
                              receiving_drugs_event_name='Received_NMF_Treatment')


def add_sapone_nmf_hs(camp, hs_df, nmf_df):
    # if no NMF rate is specified, assume all age groups have 0.0038 probability each day
    if nmf_df.empty:
        nmf_df = pd.DataFrame({'U5_nmf': [0.0038], 'adult_nmf': [0.0038]})
    elif nmf_df.shape[0] != 1:
        warnings.warn('The NMF dataframe has more than one row. Only values in the first row will be used.')
    nmf_row = nmf_df.iloc[0]

    # apply the health-seeking rate for clinical malaria to NMFs
    for r, row in hs_df.iterrows():
        add_sapone_nmf_hs_from_file(camp, row, nmf_row)


def add_sapone_nmf_hs_from_file(camp, row, nmf_row):
    start_day = row['start_day']
    duration = row['duration']
    if 'drug_code' in row.index:
        drug_code = row['drug']
    else:
        drug_code = 'AL'
    if start_day == 0:  # due to dtk diagnosis/treatment configuration, a start day of 0 is not supported
        start_day = 1  # start looking for NMFs on day 1 (not day 0) of simulation
        if duration > 1:
            duration = duration - 1
    nmf_child = nmf_row['U5_nmf']
    nmf_adult = nmf_row['adult_nmf']

    # workaround for maximum duration of 1000 days is to loop, creating a new campaign every 1000 days
    separate_durations = [1000] * int(np.floor(duration/1000))  # create a separate campaign for each 1000 day period
    if (duration - np.floor(duration/1000) > 0):  # add final remaining non-1000-day duration
        separate_durations = separate_durations + [int(duration - np.floor(duration/1000) * 1000)]
    separate_start_days = start_day + np.array([0] + list(np.cumsum(separate_durations)))
    for dd in range(len(separate_durations)):
        if nmf_child > 0:
            dc(camp, 'MSAT', drug_code=drug_code, start_days=[separate_start_days[dd]],
                              target_group={'agemin': 0, 'agemax': 5},
                              coverage=nmf_child,
                              repetitions=separate_durations[dd], tsteps_btwn_repetitions=1,
                              diagnostic_type='PF_HRP2', diagnostic_threshold=5,
                              receiving_drugs_event_name='Received_NMF_Treatment')
        if nmf_adult > 0:
            dc(camp, 'MSAT', drug_code=drug_code, start_days=[separate_start_days[dd]],
                              target_group={'agemin': 5, 'agemax': 120},
                              coverage=nmf_adult,
                              repetitions=separate_durations[dd], tsteps_btwn_repetitions=1,
                              diagnostic_type='PF_HRP2', diagnostic_threshold=5,
                              receiving_drugs_event_name='Received_NMF_Treatment')


def build_standard_campaign_object(manifest):
    import emod_api.campaign as campaign
    campaign.set_schema(manifest.schema_file)
    return campaign


def build_camp(site, coord_df=None):
    """
    Build a campaign input file for the DTK using emod_api.
    Right now this function creates the file and returns the filename. If calling code just needs an asset that's fine.
    """
    # create campaign object
    if coord_df is None:
        coord_df = pd.read_csv(manifest.simulation_coordinator_path)
        coord_df = coord_df.set_index('site')

    camp = build_standard_campaign_object(manifest)
    # === INTERVENTIONS === #

    # health-seeking
    if (not pd.isna(coord_df.at[site, 'CM_filepath'])) and (not (coord_df.at[site, 'CM_filepath'] == '')):
        hs_df = pd.read_csv(manifest.input_files_path / coord_df.at[site, 'CM_filepath'])
    else:
        hs_df = pd.DataFrame()
  
    if not hs_df.empty:
        # case management for malaria
        add_sapone_hs(camp,hs_df)
    
    # NMFs
    if (not pd.isna(coord_df.at[site, 'NMF_filepath'])) and (not (coord_df.at[site, 'NMF_filepath'] == '')):
        nmf_df = pd.read_csv(manifest.input_files_path / coord_df.at[site, 'NMF_filepath'])
    else:
        nmf_df = pd.DataFrame()
    if (not pd.isna(coord_df.at[site, 'NMF_filepath'])) and (not (coord_df.at[site, 'NMF_filepath'] == '')):
        if not hs_df.empty:
            add_sapone_nmf_hs(camp, hs_df, nmf_df)
    
    # SMC
    if (not pd.isna(coord_df.at[site, 'SMC_filepath'])) and (not (coord_df.at[site, 'SMC_filepath'] == '')):
        smc_df = pd.read_csv(manifest.input_files_path / coord_df.at[site, 'SMC_filepath'])
    else:
        smc_df = pd.DataFrame()
    if not smc_df.empty:
        # smc for children under 5 (and some leak into 5 year-olds at 50% of the 0-5 coverage)
        add_smc(camp,smc_df)

    # ITNS
    itn_df = pd.DataFrame()
    if (not pd.isna(coord_df.at[site, 'ITN_filepath'])) and (not (coord_df.at[site, 'ITN_filepath'] == '')):
        if (not pd.isna(coord_df.at[site, 'ITN_age_filepath'])) and (not (coord_df.at[site, 'ITN_age_filepath'] == '')):
            if(not pd.isna(coord_df.at[site, 'ITN_season_filepath'])) and (not (coord_df.at[site, 'ITN_season_filepath'] == '')):
                itn_df = pd.read_csv(manifest.input_files_path / coord_df.at[site, 'ITN_filepath'])
                itn_age = pd.read_csv(manifest.input_files_path / coord_df.at[site, 'ITN_age_filepath'])
                itn_season = pd.read_csv(manifest.input_files_path / coord_df.at[site, 'ITN_season_filepath'])
        
    if not itn_df.empty:
        # Distribute ITNs with age- and season-based usage patterns
        add_itns(camp,itn_df,itn_age,itn_season)

    return camp

def set_simulation_scenario(simulation, site, csv_path):
    # get information on this simulation setup from coordinator csv
    coord_df = pd.read_csv(csv_path)
    coord_df = coord_df.set_index('site')

    # === set up config === #
    # simulation duration
    simulation_duration = int(coord_df.at[site, 'simulation_duration'])
    simulation.task.config.parameters.Simulation_Duration = simulation_duration
    simulation.task.config.parameters.Enable_Vital_Dynamics = 1
    demographics_filename = str(coord_df.at[site, 'demographics_filepath'])
    #print(demographics_filename)
    if demographics_filename and demographics_filename != 'nan':
        simulation.task.transient_assets.add_asset(manifest.input_files_path / demographics_filename)
        simulation.task.config.parameters.Demographics_Filenames = [demographics_filename.rsplit('/',1)[-1]]
    simulation.task.config.parameters.Age_Initialization_Distribution_Type = 'DISTRIBUTION_COMPLEX'
    

    # === set up campaigns === #
    build_camp_partial = partial(build_camp, site=site, coord_df=coord_df)
    simulation.task.create_campaign_from_callback(build_camp_partial)
 
    return {"Site": site, 'csv_path': str(csv_path)}

#Note to Tobias, should it be like this for every site?
#it could just be demog = Demographics.from_file(manifest) and then the set Equi values and set birthrate, why are those values those values
def build_demog():
    """
    This function builds a demographics input file for the DTK using emod_api.
    """
    coord_df = load_coordinator_df(characteristic=False, set_index=True)
    DS = str(coord_df.index[0])
    new_nodes = [Demog.Node(lat=1.857481477,lon=6.444799744, pop=1000, name=DS, forced_id=1)]
    demog = Demographics.MalariaDemographics(
        nodes=new_nodes,
        idref=DS,
        init_prev=0.25,
        include_biting_heterogeneity=True,
    )
    demog.SetEquilibriumVitalDynamics(CrudeRate(38.0))
    demog.SetEquilibriumAgeDistFromBirthAndMortRates(CrudeRate(38.0), CrudeRate(38.0))
    demog.SetBirthRate(CrudeRate(38.0 * 1000))

    return demog

set_simulation_scenario_for_matched_site = partial(set_simulation_scenario, csv_path=manifest.simulation_coordinator_path)
set_simulation_scenario_for_characteristic_site = partial(set_simulation_scenario, csv_path=manifest.sweep_sim_coordinator_path)


def add_smc(camp,smc_df):
    for r in range(len(smc_df)):                                                                             
        dc(camp, campaign_type="MDA", drug_code=str(smc_df['drug_code'][r]),                   
                          start_days=[int(smc_df['start_day'][r])],                                         
                          repetitions=1,                                                                    
                          coverage=float(smc_df['coverage'][r]),                                            
                          target_group={'agemin': float(smc_df['agemin'][r]), 'agemax': float(smc_df['agemax'][r])},                                       
                          receiving_drugs_event_name="Received_SMC")                                        


def add_itns(camp,itn_df,itn_age,itn_season):
    itn_seasonal_usage = {"Times": list(itn_season['season_time']),
                          "Values":list(itn_season['season_usage'])}
    for year in itn_df['year']:                                                                                
        sub_df = itn_df[itn_df['year']==year].reset_index()                                                    
        itn_discard_config = {"Expiration_Period_Distribution": "WEIBULL_DISTRIBUTION",                        
                              "Expiration_Period_Kappa": float(sub_df['discard_k'][0]),                        
                              "Expiration_Period_Lambda": float(sub_df['discard_l'][0])}                       
        itn_age_year = itn_age[itn_age['year']==year]                                                          
        itn_age_bins = itn_age_year['age']                                                                     
        itn_age_usage = itn_age_year['age_usage']                                                              
        add_scheduled_usage_dependent_bednet(camp, intervention_name = "UsageDependentBednet",                   
                                             start_day = int(sub_df['start_day'][0]),                      
                                             demographic_coverage = float(sub_df['coverage'][0]),          
                                                   killing_initial_effect = float(sub_df['kill_effect'][0]),    
                                                   killing_decay_time_constant = int(sub_df['kill_decay'][0]),   
                                                   blocking_initial_effect = float(sub_df['block_effect'][0]),   
                                                   blocking_decay_time_constant=int(sub_df['block_decay'][0]),   
                                             age_dependence = {"Times": list(itn_age_bins),                
                                                               "Values": list(itn_age_usage)},             
                                             seasonal_dependence = itn_seasonal_usage,                     
                                             discard_config = itn_discard_config)
        

def get_comps_id_filename(site: str, level: int = 0):
    folder_name = manifest.comps_id_folder
    if level == 0:
        file_name = folder_name / (site + '_exp_submit')
    elif level == 1:
        file_name = folder_name / (site + '_exp_done')
    elif level == 2:
        file_name = folder_name / (site + '_analyzers')
    else:
        file_name = folder_name / (site + '_download')
    return file_name.relative_to(manifest.CURRENT_DIR).as_posix()

    
def add_calib_param_func(simulation, calib_params, sets, hab_base = 1e8, const_base = 1e6):
    X = calib_params[calib_params['param_set'] == sets]
    X = X.reset_index(drop=True)
    #Temperature Shift    
    simulation.task.config.parameters.Air_Temperature_Offset = float(X['emod_value'][0])      # hard-coded, assumed parameter locations
    simulation.task.config.parameters.Land_Temperature_Offset = float(X['emod_value'][0])
    #Vectors
    coord_df = load_coordinator_df(characteristic=False, set_index=True)
    vdf = pd.read_csv(os.path.join(manifest.input_files_path,coord_df['vector_filepath'].iloc[0]))
    s = [species for species in vdf['species']]
    for r in range(len(s)):
        habitat1 = dfs.schema_to_config_subnode(
            manifest.schema_file, ["idmTypes", "idmType:VectorHabitat"]
        )
        habitat1.parameters.Habitat_Type = "CONSTANT"
        malaria_config.set_species_param(
            simulation.task.config, vdf['species'][r], "Habitats", habitat1.parameters, overwrite=True
        )
        #habitat1.parameters.Max_Larval_Capacity = int(const_base * (vdf['fraction'][r] * vdf['constant'][r]) * float(X['emod_value'][1]))

        habitat2 = dfs.schema_to_config_subnode(
            manifest.schema_file, ["idmTypes", "idmType:VectorHabitat"]
        )
        habitat2.parameters.Habitat_Type = "TEMPORARY_RAINFALL"
        malaria_config.set_species_param(simulation.task.config, vdf['species'][r], "Habitats", habitat2.parameters, overwrite=False)
        #habitat2.parameters.Max_Larval_Capacity = int(hab_base * (vdf['fraction'][r] * vdf['temp_rain'][r]) * float(X['emod_value'][2]))
        
        habitat3 = dfs.schema_to_config_subnode(
            manifest.schema_file, ["idmTypes", "idmType:VectorHabitat"]
        )
        habitat3.parameters.Habitat_Type = "WATER_VEGETATION"
        malaria_config.set_species_param(simulation.task.config, vdf['species'][r], "Habitats", habitat3.parameters, overwrite=False)
        #habitat3.parameters.Max_Larval_Capacity = int(hab_base * (vdf['fraction'][r] * vdf['water_veg'][r]) * float(X['emod_value'][3]))
        
        malaria_config.set_max_larval_capacity(
            simulation.task.config, vdf['species'][r], "CONSTANT", const_base * (vdf['fraction'][r] * vdf['constant'][r]) * float(X['emod_value'][1])
        )
        malaria_config.set_max_larval_capacity(
            simulation.task.config, vdf['species'][r], "TEMPORARY_RAINFALL", hab_base * (vdf['fraction'][r] * vdf['temp_rain'][r]) * float(X['emod_value'][2])
        )
        malaria_config.set_max_larval_capacity(
            simulation.task.config, vdf['species'][r], "WATER_VEGETATION", hab_base * (vdf['fraction'][r] * vdf['water_veg'][r]) * float(X['emod_value'][3])
        )
    return {'Sample_ID':sets}

def load_coordinator_df(characteristic=False, set_index=True):
    csv_file = manifest.sweep_sim_coordinator_path if characteristic else manifest.simulation_coordinator_path
    coord_df = pd.read_csv(csv_file)
    if set_index:
        coord_df = coord_df.set_index('site')
    return coord_df


def get_suite_id():
    if os.path.exists(manifest.suite_id_file):
        with open(manifest.suite_id_file, 'r') as id_file:
            suite_id = id_file.readline()
        return suite_id
    else:
        return 0
      
def generate_demographics():
  
    coord_df=load_coordinator_df(characteristic=False, set_index=True)
    site = str(coord_df.index[0])
    latitude=coord_df.at[site,'lat']
    longitude=coord_df.at[site,'lon']
    population=coord_df.at[site,'pop']
    prev0 = coord_df.at[site,'prev0']
    BR = coord_df.at[site,'birth_rate']

    new_nodes = [Demog.Node(lat=float(latitude),
                            lon=float(longitude),
                            pop=int(population),
                            name=str(site),
                            forced_id=1)]

    demog = Demographics.MalariaDemographics(nodes=new_nodes,
                                             idref=str(site),
                                             init_prev=float(prev0),
                                             include_biting_heterogeneity=True)

    print("Setting Equilibrium Vital Dynamics")
    demog.SetEquilibriumVitalDynamics(CrudeRate(float(BR)))

    print("Getting Equilibrium Age Distribution")
    demog.SetEquilibriumAgeDistFromBirthAndMortRates(CrudeRate(float(BR)),
                                                     CrudeRate(float(BR)))

    print("Amending Birth Rate")
    demog.SetBirthRate(CrudeRate(float(BR) * 1000))
    #print(demog.__dict__)
    with open(f"../simulation_inputs/demographics_files/{site}_demographics.json", "w") as outfile:
        json.dump(demog.to_dict(), outfile, indent=3, sort_keys=True)


    print(f"Saved to ../simulation_inputs/demographics_files/{site}_demographics.json")
    return demog
  

def extract_climate(flatten_temp=True):
  
    import time
    from emodpy_malaria.weather import (generate_weather, weather_to_csv, WeatherVariable, 
                                        csv_to_weather)
    coord_df=load_coordinator_df(characteristic=False, set_index=True)
    # ---| Request weather files |---
    site = str(coord_df.index[0])
    start_yr = coord_df.at[site,'climate_start_year']
    length = coord_df.at[site,'climate_year_dur']

    extractdir = '../simulation_inputs/tmp/'
    outdir = os.path.join('../simulation_inputs/site_climate', site)

    if not os.path.exists(extractdir):
        os.makedirs(extractdir)
    
    weather_dir = extractdir
    startdate = start_yr * 1000 + 1
    enddate = (start_yr + length - 1) * 1000 + 365
    
    wr = generate_weather(platform="Calculon",
                          site_file=manifest.simulation_coordinator_path,
                          start_date=startdate,
                          end_date=enddate,
                          node_column="nodes",
                          id_reference=site,
                          local_dir=weather_dir,
                          force=True)
    time.sleep(10)

    df, wa = weather_to_csv(weather_dir)
    weather_columns = {
        WeatherVariable.AIR_TEMPERATURE: "airtemp",
        WeatherVariable.RELATIVE_HUMIDITY: "humidity",
        WeatherVariable.RAINFALL: "rainfall"
    }
    weather_filenames = {
        WeatherVariable.AIR_TEMPERATURE: "air_temperature_daily.bin",
        WeatherVariable.RELATIVE_HUMIDITY: "relative_humidity_daily.bin",
        WeatherVariable.RAINFALL: "rainfall_daily.bin"
    }
    
    # Remove extra day in 2016
    df = df[df.steps != 1096]
    df.steps = [x if x < 1096 else x - 1 for x in df.steps]
    df1 = df.copy()
    print(df1['airtemp'])
    # Set constant air temperature to the mean
    print("flattening temp")
    if flatten_temp:
        airtempMean = df1["airtemp"].mean()
        df1.loc[:, "airtemp"] = airtempMean
    print(df1['airtemp'])
    csv_to_weather(df1, attributes=wa, weather_columns=weather_columns,
                   weather_dir=outdir,
                   weather_file_names=weather_filenames)
