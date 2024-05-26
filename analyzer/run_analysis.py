import analyzer
import analyzer.core as ac
import analyzer.datasets as ds

def runAnalysisOnSamples(modules, samples, sample_manager, dask_schedd_address=None, dataset_directory="datasets",step_size=75000,delayed=True,):

    import analyzer.modules

    cache = {}
    if dask_schedd_address:
        print(f"Connecting client to scheduler at {dask_schedd_address}")
        client = Client(dask_schedd_address)
        transferAnalyzerToClient(client)
    else:
        client = None
        print("No scheduler address provided, running locally")

    sample_manager = ds.SampleManager()
    sample_manager.loadSamplesFromDirectory("datasets")
    print(f"Creating analyzer using {len(modules)} modules")
    print(modules)
    analyzer = ac.Analyzer(modules, cache)

