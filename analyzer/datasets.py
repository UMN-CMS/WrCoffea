import os
import yaml

class AnalyzerInput:
    def __init__(self, dataset_name: str, fill_name: str):
        self.dataset_name = dataset_name
        self.fill_name = fill_name

class SampleManager:
    def __init__(self):
        self.sets = {}
        self.collections = {}

    def loadSamplesFromDirectory(self, directory):
        print("In datasets.py")
        files = [file for file in os.listdir(directory) if file.endswith(".yaml")]
        file_contents = {}

        print(f"List of yaml files: {files}\n")

        for file in files:
            print(f"Looking at file {file}\n")
            file_path = os.path.join(directory, file)
            print(f"File path is {file_path}\n")
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
                print(f"The file data is {data}\n")
                file_contents[file] = data
                
                for item in data:
                    if item.get("type", "") == "set" or "files" in item:
                        print(f"item {item}\n")
                        sample_set = SampleSet(item["name"], item["mc_campaign"], item["x_sec"], item["n_events"], item["lumi"], item["files"])
                        sample_set.from_dict(item)
                        self.sets[sample_set.name] = sample_set

        print(f"self.sets: {self.sets}")

class SampleSet:

    def __init__(self, name, mc_campaign, x_sec, n_events, lumi, files):
        self.name = name
        self.mc_campaign = mc_campaign
        self.x_sec = x_sec
        self.n_events = n_events
        self.lumi = lumi
        self.files = files

    def from_dict(self, data):
        name = data["name"]
        lumi = data.get("lumi")
        x_sec = data.get("x_sec")
        n_events = data.get("n_events")
        mc_campaign = data.get("mc_campaign")

        files = []
        if "files" in data:
            for file_data in data["files"]:
                print(f"file_data: {file_data}")
                files.append(SampleFile.from_dict(file_data))
        
        return SampleSet(
            name,
            lumi,
            x_sec,
            n_events,
            files,
            mc_campaign,
        )

class SampleFile:
    def __init__(self, paths=None):
        # Initialize paths attribute
        self.paths = paths if paths is not None else []

    @staticmethod
    def from_dict(data):
        # Check if data is a list
        if isinstance(data, list):
            return SampleFile(data)
        else:
            return SampleFile([data])

    def get_root_dir(self):
        # Return the root directory
        return "Events"

    def get_file(self):
        # Return the first path in paths
        return self.paths[0] if self.paths else None
