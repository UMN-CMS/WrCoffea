from analyzer.datasets import AnalyzerInput, SampleManager
from  .inputs import DatasetPreprocessed
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema

def get_events(arg, known_form=None, cache=None):
    if isinstance(arg, AnalyzerInput):
        # If arg is an AnalyzerInput object
        ds_preprocessed = DatasetPreprocessed.fromDatasetInput(arg)
        files = ds_preprocessed.coffea_dataset_split["files"]
        # Recursive call to get_events with the files
        return get_events(files)

    elif isinstance(arg, str):
        # If arg is a string
        if "/" in arg:
            # If arg is a file path
            return get_events({arg: "Events"})
        else:
            # If arg is a sample name
            sample_manager = SampleManager()
            sample_manager.loadSamplesFromDirectory(arg)
            sample = sample_manager.getSet(arg)
            # Recursive call to get_events with the sample's AnalyzerInput
            return get_events(sample.getAnalyzerInput())

    # Default case
    events, report = NanoEventsFactory.from_root(
        arg,
        schemaclass=NanoAODSchema,
        uproot_options=dict(
            allow_read_errors_with_report=True,
        ),
        known_base_form=known_form,
        persistent_cache=cache,
    ).events()
    return events, report

