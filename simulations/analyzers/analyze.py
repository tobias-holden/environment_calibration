import argparse
import os

from idmtools.analysis.analyze_manager import AnalyzeManager
from idmtools.core import ItemType

from analyzers.analyzer_collection import (
    EventReporterAnalyzer,
    MonthlyPfPRAnalyzer,
    AnnualPfPRAnalyzer,
    InsetChartAnalyzer,
    EventReporterSummaryAnalyzer,
    NodeDemographicsAnalyzer,
    VectorStatsAnalyzer
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", dest="site", type=str, required=True)
    parser.add_argument("--expid", dest="expid", type=str, required=True)

    return parser.parse_args()


def analyze_experiment(platform, expid, wdir, report_start_day, simulation_duration):
    if not os.path.exists(wdir):
        os.makedirs(wdir)

    analyzers = []
    # custom analyzers
    sweep_variables = ['Run_Number', 'Sample_ID']
    #it would be useful to check to make sure that the sweep_variables selected are the only ones necessary.
    #additionally, I would check to make sure that insetchartanalyzer doesn't need channels defined

    # analyzers.append(EventReporterSummaryAnalyzer(sweep_variables=sweep_variables,
    #                                              working_dir=wdir, 
    #                                              time_cutoff=0,
    #                                              event_list=["Received_Treatment", "Bednet_Using","Bednet_Got_New_One","Bednet_Discarded", "Received_SMC"],
    #                                              output_filename="event_counts"))
    # analyzers.append(NodeDemographicsAnalyzer(sweep_variables=sweep_variables,
    #                                           working_dir=wdir,
    #                                           output_filename="age_population",
    #                                           time_cutoff=0))

    analyzers.append(InsetChartAnalyzer(sweep_variables=sweep_variables,
                                       working_dir=wdir,
                                       start_day=int(report_start_day/365),
                                       channels=["PCR Parasite Prevalence", "Air Temperature","Daily EIR"]))

    # analyzers.append(VectorStatsAnalyzer(sweep_variables=sweep_variables,
    #                                      working_dir=wdir,
    #                                      start_time=int(report_start_day/365),
    #                                      end_time=int(simulation_duration/365),
    #                                      ))

    # Don't change these - used for fitting #
    analyzers.append(MonthlyPfPRAnalyzer(sweep_variables=sweep_variables,
                                        working_dir=wdir,
                                        start_year=int(report_start_day/365),
                                        end_year=int(simulation_duration/365)))
    analyzers.append(AnnualPfPRAnalyzer(sweep_variables=sweep_variables,
                                        working_dir=wdir,
                                        start_year=int(report_start_day/365),
                                        end_year=int(simulation_duration/365)))                                    
    # analyzers.append(EventReporterAnalyzer(sweep_variables=sweep_variables,
    #                                       working_dir=wdir,
    #                                       time_cutoff=int(report_start_day),
    #                                       event_list=["Received_Treatment"],
    #                                       output_filename="events"))
    
    manager = AnalyzeManager(platform=platform,
                             configuration={},
                             ids=[(expid, ItemType.EXPERIMENT)],
                             analyzers=analyzers,
                             partial_analyze_ok=True,
                             max_workers=16)
    
    manager.analyze()


if __name__ == "__main__":
    
    from idmtools.core.platform_factory import Platform
    import manifest
    args = parse_args()
    platform = Platform('SLURM_LOCAL', job_directory=manifest.job_directory)
    outdir = args.site
    analyze_experiment(platform, 
                       args.expid,
                       os.path.join(manifest.output_dir, outdir))
