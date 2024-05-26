from .org import namesToModules, sortModules

class Analyzer:
    def __init__(self, modules, cache):
        self.modules = self.__createAndSortModules(*modules)
        print(
            "Will run modules in the following order:\n"
            + "\n".join(f"\t{i+1}. {x.name}" for i, x in enumerate(self.modules))
        )
        self.cache = cache
        self.__dataset_ps = {}
        self.__run_reports = {}

    def __createAndSortModules(self, *module_names):
        print("In createAndSortModules")
        m = namesToModules(module_names)
#        t = generateTopology(m)
        modules = sortModules(m)
        return modules
