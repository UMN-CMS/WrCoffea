import numpy as np
import awkward as ak
def createObjects(events):

    elecs = events.Electron
    muons = events.Muon
    jets = events.Jet

    good_elecs = elecs[(elecs.pt > 53) & (np.abs(elecs.eta) < 2.4) & (elecs.cutBased_HEEP)]
    good_muons = muons[(muons.pt > 53) & (np.abs(muons.eta) < 2.4) & (muons.highPtId == 2) & (muons.tkRelIso < 0.1)]
    good_jets = jets[(jets.pt > 40) & (np.abs(jets.eta) < 2.4) & (jets.isTightLeptonVeto)]

    leptons = ak.with_name(ak.concatenate((good_elecs, good_muons), axis=1), 'PtEtaPhiMCandidate')
    leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]

    events["good_elecs"] = good_elecs
    events["good_muons"] = good_muons
    events["good_jets"] = good_jets
    events["leptons"] = leptons

    return events
