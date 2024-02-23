# environment_calibration

## Background

Our goal is to adapt a botorch multi-objective Bayesian optimization workflow to select EMOD habitat & climate parameters to fit reference site seasonality.

Status Summary
||Tested|*Not Yet Tested*|
|:-:|:-|:-|
|Interventions|treatment-seeking, non-malaria fever treatment,<br>SMC, bednets (season- and age-dependent)|IRS, Campaigns (MSAT/MDA)|
|Vectors|*arabiensis*, *gambiae*, *funestus*||
|Habitats|constant, temporary rainfall, water vegetation|brackish swamp, population-dependent|

## Methods

## Input Parameter Space

| parameter | min | max | transformation |
|:---------:|:---:|:---:|:--------------:|
| Temperature Shift | -5 | 5 | none |
| Constant Habitat Multiplier | -4 | 4 | log |
| Temporary Rainfall Habitat Multiplier | -4 | 4 | log |
| Water Vegetation Habitat Multiplier | -4 | 4 | log |

## Objectives

| Objective | weight | Data source | EMOD output |
|:---------:|:------:|:-----------:|:-----------:|
| Normalized Monthly Incidence | 1.0 |  | "Received Treatment" events over age 5 |
| Average Annual Incidence | 0.1 |  | "Received Treatment" events over age 5 |
| Monthly Prevalence | 0.1 |  | InsetChart |
| Monthly EIR Threshold | 10.0 |  | InsetChart |



### Blood Loss

## Sites

### Sapone

#### Climate

#### Vectors

#### Treatment

#### SMC

#### Bednets

### Nanoro

#### Climate

#### Vectors

#### Treatment

#### SMC

#### Bednets
