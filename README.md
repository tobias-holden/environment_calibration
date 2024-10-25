# environment_calibration

## Background

Our goal is to adapt a botorch multi-objective Bayesian optimization workflow to select EMOD habitat & climate parameters to fit reference site seasonality.

Status Summary

|       | Tested | *Not Yet Tested*     |
|:-----:|:-------|:---------------------|
| Interventions | treatment-seeking, non-malaria fever treatment, SMC, bednets (season- and age-dependent) | IRS, Campaigns (MSAT/MDA), vaccDrugSMC |
|    Vectors    | *arabiensis*, *gambiae*, *funestus*                                                      |                                        |
|   Habitats    | constant, temporary rainfall, water vegetation                                           | linear spline                          |

## Instructions

### 1 Create Virtual Environment

To start, create a virtual environment containing botorch, idmtools, emodpy, and other required packages.

For Quest users, you can build an environment based off of the existing \<emodpy-torch\> environment.

``` bash
# clone conda-compatible parts of existing virtual environment conda create --prefix <path/to/env> --name <YOUR_ENVIRONMENT> --clone /projects/b1139/environments/emodpy-torch
```

-   The suggested path/to/env for Quest users is /projects/b1139/environments

    -   Otherwise, the environment will be created in your current working directory, in a new folder called .conda/envs/

-   **Make sure YOUR_ENVIRONMENT is a unique name that doesn't already exist in the folder**

The previous step clones all conda-compatible parts of the virtual environment. It can take a while, but you can expect some terminal output along the way...

``` bash
Source:      /projects/b1139/environments/emodpy-torch Destination: /projects/b1139/environments/.conda/envs/<YOUR_ENVIRONMENT> Packages: 50 Files: 26196 Downloading and Extracting Packages Preparing transaction: done Verifying transaction: done Executing transaction: done # By downloading and using the cuDNN conda packages, you accept the terms and conditions # of the NVIDIA cuDNN EULA - https://docs.nvidia.com/deeplearning/cudnn/sla/index.html # # To activate this environment, use #     $ conda activate test_torch # To deactivate an active environment, use # #     $ conda deactivate
```

Because some of the packages in the `emodpy-torch` environment were installed using `pip`, they might not make it through the conda clone step. To add them:

``` bash
# Activate your new virtual environment source activate <path/to/env>/<YOUR_ENVIRONMENT> # ex. /projects/b1139/environments/my_environment  # pip install from requirements.txt pip install -r /projects/b1139/environments/emodpy-torch/requirements.txt
```

### 2. Customize simulation inputs

1.  Describe reference site
    -   **simulation_coordinator.csv**
    -   Related .csv files for *vectors* and *interventions*

|               option                |                    value                    |              description               |
|:------------------:|:-----------------:|:-------------------------------:|
|                node                 |                      1                      |                                        |
|                site                 |                   Nanoro                    |                                        |
|                 lat                 |                    12.68                    |                                        |
|                 lon                 |                    -2.19                    |                                        |
|         climate_start_year          |                    2010                     |                                        |
|          climate_year_dur           |                     10                      |                                        |
|                 pop                 |                    1000                     |                                        |
|             birth_rate              |                     38                      |                                        |
|                prev0                |                     0.2                     |                                        |
|            include_site             |                    TRUE                     |                                        |
|           run_script_name           |                 run_sims.py                 |                                        |
|                nSims                |                      1                      |                                        |
|        simulation_start_year        |                    2000                     |                                        |
|          simulation_years           |                     20                      |                                        |
|        demographics_filepath        | demographics_files/Nanoro_demographics.json |                                        |
|            NMF_filepath             |  nonmalarial_fevers/nmf_rates_generic.csv   |        blank if not applicable         |
|             CM_filepath             |        cm/Nanoro_case_management.csv        |        blank if not applicable         |
|            SMC_filepath             |                                             |        blank if not applicable         |
|            ITN_filepath             |             itn/Nanoro_ITN.csv              |        blank if not applicable         |
|          ITN_age_filepath           |               itn/ITN_age.csv               |        blank if not applicable         |
|         ITN_season_filepath         |             itn/ITN_season.csv              |        blank if not applicable         |
|           vector_filepath           |             vectors/vectors.csv             |                                        |
|        prevalence_comparison        |                    TRUE                     |                                        |
|   prevalence_comparison_reference   |          pcr_prevalence_AllAge.csv          |                                        |
|   prevalence_comparison_frequency   |                   monthly                   |      """monthly"" or ""annual"""       |
|  prevalence_comparison_diagnostic   |                     PCR                     | """PCR"" or ""Microscopy"" or ""RDT""" |
| include_AnnualMalariaSummaryReport  |                    TRUE                     |                                        |
|        incidence_comparison         |                    TRUE                     |                                        |
|   incidence_comparison_reference    |      routine_incidence_by_district.csv      |                                        |
|   incidence_comparison_frequency    |                   monthly                   |      """monthly"" or ""annual"""       |
|     incidence_comparison_agebin     |                     100                     |                                        |
| include_MonthlyMalariaSummaryReport |                    TRUE                     |                                        |

2.  run **create_files.py** to generate climate and demographics files.

    -   Files created inside simulation_inputs/:
        -   demographics_files/*site*\_demographics.json
        -   site_climate/*site*/...
    -   If you already have files:
        -   supply the path to the desired demographics file inside simulation_coordinator.csv 'demographics_filepath' row
        -   copy climate files into folder site_climate/*site*/

3.  Define input parameter sampling space

    -   **test_parameter_key.csv**

        |               parameter               | min | max | transformation |
        |:-------------------------------------:|:---:|:---:|:--------------:|
        |           Temperature Shift           | -5  |  5  |      none      |
        |      Constant Habitat Multiplier      | -4  |  4  |      log       |
        | Temporary Rainfall Habitat Multiplier | -4  |  4  |      log       |
        |  Water Vegetation Habitat Multiplier  | -4  |  4  |      log       |

4.  Refine scoring system

    -   **my_weights.csv**

        |    objective     | weight |                                     |
        |:----------------:|:------:|-------------------------------------|
        |   shape_score    | 0.001  | *Normalized monthly incidence*      |
        | intensity_score  |  0.1   | *Average annual clinical incidence* |
        | prevalence_score |  0.1   | *Monthly all-age prevalence*        |
        |    eir_score     |  10.0  | *EIR threshold*                     |

5.  Set up calibration scheme

    -   **calib_coordinator.csv**

        | init_size | init_batches | batch_size | max_eval | failure_limit |
        |:---------:|:------------:|:----------:|:--------:|:-------------:|
        |   1000    |      1       |    200     |   5000   |       2       |

6.  edit **run_calib.py** with updated experiment name

7.  run **`sbatch sbatch_run_calib.sh`**

8.  run **post_calibration_plots.Rmd** to generate additional plots *(optional)*

## Methods
