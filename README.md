# environment_calibration

## Instructions


1. Describe reference site
    * **simulation_coordinator.csv**
    * Related .csv files for *vectors* and *interventions*
      
2. run **create_files.py** to generate climate and demographics files.  

    - If you already have files, just supply the path to them inside simulation_coordinator.csv 'demographics_filepath' column.      

3. Define input parameter sampling space  
    * **test_parameter_key.csv**
    
        | parameter | min | max | transformation |
        |:---------:|:---:|:---:|:--------------:|
        | Temperature Shift | -5 | 5 | none |
        | Constant Habitat Multiplier | -4 | 4 | log |
        | Temporary Rainfall Habitat Multiplier | -4 | 4 | log |
        | Water Vegetation Habitat Multiplier | -4 | 4 | log |
      
4. Refine scoring system
    * **objectives.csv**
      
        | Objective | weight | 
        |:---------:|:------:|
        | Normalized Monthly Incidence | 1.0 | 
        | Average Annual Incidence | 0.1 | 
        | Monthly Prevalence | 0.1 | 
        | Monthly EIR Threshold | 10.0 |

5. Set up calibration scheme
     * **calib_coordinator.csv**
       
        | init_size | init_batches | batch_size | max_eval | failure_limit |
        |:---------:|:------------:|:----------:|:--------:|:--------------:|
        | 1000 | 1 | 200 | 5000 | 2 |

6. edit **run_calib.py** with updated experiment name

7. run **sbatch sbatch_run_calib.sh**

8. run **calib_details.py** to generate additional plots *(optional)*

## Background

Our goal is to adapt a botorch multi-objective Bayesian optimization workflow to select EMOD habitat & climate parameters to fit reference site seasonality.

Status Summary
||Tested|*Not Yet Tested*|
|:-:|:-|:-|
|Interventions|treatment-seeking, non-malaria fever treatment,<br>SMC, bednets (season- and age-dependent)|IRS, Campaigns (MSAT/MDA)|
|Vectors|*arabiensis*, *gambiae*, *funestus*||
|Habitats|constant, temporary rainfall, water vegetation|linear spline|


## Methods





