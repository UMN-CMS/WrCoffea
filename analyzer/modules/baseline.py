import awkward as ak
from coffea.analysis_tools import PackedSelection

def createSelection(events):
    """Baseline selection for the analysis.
        Applies the following selection:
            - 2 leptons
            - 2 <= nJets
            - leptons[0].pt > 60
            - mlljj > 800t
            - flavor selections
    """

    good_elecs = events.good_elecs
    good_muons = events.good_muons
    good_jets = events.good_jets

    leptons = events.leptons

    padded_leptons = ak.pad_none(leptons, 2, axis=1)
    padded_jets = ak.pad_none(good_jets, 2, axis=1)

    mlljj = ak.fill_none((padded_leptons[:, 0] + padded_leptons[:, 1] + padded_jets[:, 0] + padded_jets[:, 1]).mass, False)
    mll = ak.fill_none((padded_leptons[:, 0] + padded_leptons[:, 1]).mass, False)

    dr_jl_min = ak.fill_none(ak.min(padded_jets[:,:2].nearest(padded_leptons).delta_r(padded_jets[:,:2]), axis=1), False)
    dr_j1j2 = ak.fill_none(padded_jets[:,0].delta_r(padded_jets[:,1]), False)
    dr_l1l2 = ak.fill_none(padded_leptons[:,0].delta_r(padded_leptons[:,1]), False)

    selection = PackedSelection()
    selection.add("exactly2l", ((ak.num(good_elecs) + ak.num(good_muons)) == 2))
    selection.add("atleast2j", (ak.num(good_jets) >= 2))
    selection.add("leadleppt60", ((ak.any(good_elecs.pt > 60, axis=1)) | (ak.any(good_muons.pt > 60, axis=1))))
    selection.add("mlljj>800", (mlljj > 800))
    selection.add("dr>0.4", (dr_jl_min > 0.4) & (dr_j1j2 > 0.4) & (dr_l1l2 > 0.4))
    selection.add("eejj", ((ak.num(good_elecs) == 2) & (ak.num(good_muons) == 0)))
    selection.add("mumujj", ((ak.num(good_elecs) == 0) & (ak.num(good_muons) == 2)))
    selection.add("emujj", ((ak.num(good_elecs) == 1) & (ak.num(good_muons) == 1)))
    selection.add("60mll150", ((mll > 60) & (mll < 150)))
    selection.add("150mll400", ((mll > 150) & (mll < 400)))
    selection.add("400mll", (mll > 400))

    return selection

