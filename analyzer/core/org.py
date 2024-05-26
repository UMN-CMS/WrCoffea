def toSet(x):
    if iterableNotStr(x):
        return set(x)
    else:
        return (x,)

class AnalyzerModule:
    def __init__(self,name,function,depends_on=None,categories="main",after=None,always=False,documentation=None,):
        self.name = name
        self.function = function
        self.depends_on = toSet(depends_on) if depends_on else set()
        self.categories = toSet(categories) if categories else set()
        self.always = always
        self.documenation = documentation

#modules = {}

def generateTopology(module_list):
    mods = [x.name for x in module_list]
    mods.extend([x.name for x in modules.values() if x.always])
    for name in mods:
        module = modules[name]


def namesToModules(module_list):
    result = []
    for x in module_list:
        print(x)
        result.append(modules[x])
    return [modules[x] for x in module_list]

def sortModules(module_list):
    return namesToModules(ret)
