import argparse
import analyzer.datasets as ds
import analyzer.run_analysis as ra

from pathlib import Path

def runCli():
    parser = argparse.ArgumentParser(prog="WrAnalyzer")
    subparsers = parser.add_subparsers()
    addSubparserRun(subparsers)

    args = parser.parse_args()

    return args

def addSubparserRun(subparsers):
    subparser = subparsers.add_parser("run", help="Run analyzer over some collection of samples")
    subparser.add_argument("-o", "--output", required=True, type=Path, help="Output data path.")
    run_mode = subparser.add_mutually_exclusive_group()
    run_mode.add_argument("-a","--scheduler-address",type=str,help="Address of the scheduler to use for dask",)
    run_mode.add_argument("--no-delayed",action="store_true",default=False,help="Do not use dask, instead run synchronously. Good for testing.",)
    subparser.add_argument("-m", "--modules",type=str,nargs="+",help="List of modules to execute.",metavar="",)
    subparser.add_argument("-s","--samples",type=str,nargs="+",help="List of samples to run over",metavar="",)
    subparser.add_argument("--step-size",default=100000,type=int,help="Number of events per chunk",)
    subparser.set_defaults(func=handleRunAnalysis)

def handleRunAnalysis(args):
    print("Handling run analysis")
    dataset_path = "datasets"
    sample_manager = ds.SampleManager()
    sample_manager.loadSamplesFromDirectory("datasets")
    ret = ra.runAnalysisOnSamples(
        args.modules,
        args.samples,
        sample_manager,
        dask_schedd_address=args.scheduler_address,
        delayed=not args.no_delayed,
        step_size=args.step_size,
    )
