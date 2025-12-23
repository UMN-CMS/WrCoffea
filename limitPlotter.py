#!/usr/bin/env python3
from ROOT import *
import sys, os

gROOT.SetBatch(True)

if len(sys.argv) < 5:
    print("Usage: python plot_expected_limits_filtered.py <combine_output_2jet.root> <combine_output_3jet.root> <combine_output_narrow.root> <theory_xsec.txt> <output_name.pdf> <MN_GeV>")
    sys.exit(1)

combine_file = sys.argv[1]
combine_file_3jet = sys.argv[2]
combine_file_narr = sys.argv[3]
theory_file = sys.argv[4]
outfile = sys.argv[5]
target_mN = float(sys.argv[6])   # e.g. 100

# ======================================================
# Read theoretical cross section file
# Columns: MW MN Xsec(pb) Unc_Xsec(pb)
# ======================================================
masses, xsecs, xsec_unc = [], [], []

with open(theory_file, 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 3:
                MW = float(parts[0])
                if MW in [ 6000]:
                    continue
                MN = float(parts[1])
                X = float(parts[2])* 1000.0       # pb â fb
                dX = float(parts[2]) * 100.0
                target_mN = MW/2
                if abs(MN - target_mN) < 1e-3:     # match this N mass
                    print(MW, MN, X, target_mN)
                    masses.append(MW)
                    xsecs.append(X)
                    #xsec_unc.append(dX)
masses, xsecs = zip(*sorted(zip(masses, xsecs)))
masses = list(masses)
xsecs = list(xsecs)                    
if not masses:
    print(f"[ERROR] No entries found in theory file for MN = {target_mN} GeV")
    sys.exit(1)

# ======================================================
# Read expected limits from Combine output
# ======================================================

infile = TFile(combine_file, "READ")
tree = infile.Get("limit")

infile_3jet = TFile(combine_file_3jet, "READ")
tree_3jet = infile_3jet.Get("limit")

infile_narr = TFile(combine_file_narr, "READ")
tree_narr = infile_narr.Get("limit")

branches = [b.GetName() for b in tree.GetListOfBranches()]
has_mN = any("N" in b for b in branches)

branches_3jet = [b.GetName() for b in tree_3jet.GetListOfBranches()]
has_mN_3jet = any("N" in b for b in branches_3jet)

branches_narr = [b.GetName() for b in tree_narr.GetListOfBranches()]
has_mN_narr = any("N" in b for b in branches_narr)

mass_vals = []
exp_median, exp_m1s, exp_p1s, exp_m2s, exp_p2s = {}, {}, {}, {}, {}

mass_vals_3jet = []
exp_median_3jet, exp_m1s_3jet, exp_p1s_3jet, exp_m2s_3jet, exp_p2s_3jet = {}, {}, {}, {}, {}

mass_vals_narr = []
exp_median_narr, exp_m1s_narr, exp_p1s_narr, exp_m2s_narr, exp_p2s_narr = {}, {}, {}, {}, {}

for event in tree:
    # Identify masses
    mWR = int(event.mh/10000)
    mN = getattr(event, "mN", target_mN) if has_mN else target_mN
    #print(mWR, mN, "printing particle masses")
    # Keep only target N mass
    target_mN = mWR/2
    if mWR ==8000:
        continue
    if abs(mN - target_mN) > 1e-3:
        continue
    
    q = event.quantileExpected
    if q < 0:  # skip observed
        continue

    val = event.limit #*1000  # pb â fb

    if abs(q - 0.5) < 1e-3:
        exp_median[mWR] = val
    elif abs(q - 0.16) < 1e-3:
        exp_m1s[mWR] = val
    elif abs(q - 0.84) < 1e-3:
        exp_p1s[mWR] = val
    elif abs(q - 0.025) < 1e-3:
        exp_m2s[mWR] = val
    elif abs(q - 0.975) < 1e-3:
        exp_p2s[mWR] = val

    if (mWR not in mass_vals) or (mWR!=6001) or (mWR !=8001):
        mass_vals.append(mWR)
        #print(q, val, mWR, mN)
mass_vals = sorted(list(set(mass_vals)))

for event in tree_3jet:
    # Identify masses
    mWR = int(event.mh/10000)
    mN = getattr(event, "mN", target_mN) if has_mN_3jet else target_mN
    # Keep only target N mass
    target_mN = mWR/2
    if mWR ==8000:
        continue
    if abs(mN - target_mN) > 1e-3:
        continue
    
    q = event.quantileExpected
    if q < 0:  # skip observed
        continue

    val = event.limit #*1000  # pb â fb

    if abs(q - 0.5) < 1e-3:
        exp_median_3jet[mWR] = val
    elif abs(q - 0.16) < 1e-3:
        exp_m1s_3jet[mWR] = val
    elif abs(q - 0.84) < 1e-3:
        exp_p1s_3jet[mWR] = val
    elif abs(q - 0.025) < 1e-3:
        exp_m2s_3jet[mWR] = val
    elif abs(q - 0.975) < 1e-3:
        exp_p2s_3jet[mWR] = val

    if (mWR not in mass_vals_3jet) or (mWR!=6001) or (mWR !=8001):
        mass_vals_3jet.append(mWR)
        #print(q, val, mWR, mN)
mass_vals_3jet = sorted(list(set(mass_vals_3jet)))

for event in tree_narr:
    # Identify masses
    mWR = int(event.mh/10000)
    mN = getattr(event, "mN", target_mN) if has_mN_narr else target_mN
    # Keep only target N mass
    target_mN = mWR/2
    if mWR ==8000:
        continue
    if abs(mN - target_mN) > 1e-3:
        continue
    
    q = event.quantileExpected
    if q < 0:  # skip observed
        continue

    val = event.limit #*1000  # pb â fb

    if abs(q - 0.5) < 1e-3:
        exp_median_narr[mWR] = val
    elif abs(q - 0.16) < 1e-3:
        exp_m1s_narr[mWR] = val
    elif abs(q - 0.84) < 1e-3:
        exp_p1s_narr[mWR] = val
    elif abs(q - 0.025) < 1e-3:
        exp_m2s_narr[mWR] = val
    elif abs(q - 0.975) < 1e-3:
        exp_p2s_narr[mWR] = val

    if (mWR not in mass_vals_narr) or (mWR!=6001) or (mWR !=8001):
        mass_vals_narr.append(mWR)
        #print(q, val, mWR, mN)
mass_vals_narr = sorted(list(set(mass_vals_narr)))


if not exp_median:
    print(f"[ERROR] No expected limits found for MN = {target_mN} GeV.")
    sys.exit(1)

if not exp_median_3jet:
    print(f"[ERROR] No expected limits found for MN = {target_mN} GeV.")
    sys.exit(1)

if not exp_median_narr:
    print(f"[ERROR] No expected limits found for MN = {target_mN} GeV.")
    sys.exit(1)
# ======================================================
# Build TGraphs
# ======================================================

gr_95 = TGraphAsymmErrors(len(mass_vals))
gr_68 = TGraphAsymmErrors(len(mass_vals))
gr_median = TGraph(len(mass_vals))
print(len(mass_vals), len(xsecs))

gr_95_3jet = TGraphAsymmErrors(len(mass_vals_3jet))
gr_68_3jet = TGraphAsymmErrors(len(mass_vals_3jet))
gr_median_3jet = TGraph(len(mass_vals_3jet))
print(len(mass_vals_3jet), len(xsecs))

gr_95_narr = TGraphAsymmErrors(len(mass_vals_narr))
gr_68_narr = TGraphAsymmErrors(len(mass_vals_narr))
gr_median_narr = TGraph(len(mass_vals_narr))
print(len(mass_vals_narr), len(xsecs))

for i, m in enumerate(mass_vals):
    median = exp_median.get(m, 0)
    m1s = exp_m1s.get(m, median)
    p1s = exp_p1s.get(m, median)
    m2s = exp_m2s.get(m, median)
    p2s = exp_p2s.get(m, median)
    #print(median, m)
    median =  median*xsecs[i]
    m1s = m1s*xsecs[i]
    p1s = p1s*xsecs[i]
    p2s  = p2s*xsecs[i]
    m2s = m2s*xsecs[i]
    gr_median.SetPoint(i, m, median)
    gr_68.SetPoint(i, m, median)
    gr_68.SetPointError(i, 0, 0, median - m1s, p1s - median)
    gr_95.SetPoint(i, m, median)
    gr_95.SetPointError(i, 0, 0, median - m2s, p2s - median)

factor=1

for i, m in enumerate(mass_vals_3jet):
    median = exp_median_3jet.get(m, 0)
    m1s = exp_m1s_3jet.get(m, median)
    p1s = exp_p1s_3jet.get(m, median)
    m2s = exp_m2s_3jet.get(m, median)
    p2s = exp_p2s_3jet.get(m, median)
    #print(median, m)
    median =  median*xsecs[i]/factor
    m1s = m1s*xsecs[i]/factor
    p1s = p1s*xsecs[i]/factor
    p2s  = p2s*xsecs[i]/factor
    m2s = m2s*xsecs[i]/factor
    gr_median_3jet.SetPoint(i, m, median)
    gr_68_3jet.SetPoint(i, m, median)
    gr_68_3jet.SetPointError(i, 0, 0, median - m1s, p1s - median)
    gr_95_3jet.SetPoint(i, m, median)
    gr_95_3jet.SetPointError(i, 0, 0, median - m2s, p2s - median)

factor=4

for i, m in enumerate(mass_vals_narr):
    median = exp_median_narr.get(m, 0)
    m1s = exp_m1s_narr.get(m, median)
    p1s = exp_p1s_narr.get(m, median)
    m2s = exp_m2s_narr.get(m, median)
    p2s = exp_p2s_narr.get(m, median)
    #print(median, m)
    median =  median*xsecs[i]/factor
    m1s = m1s*xsecs[i]/factor
    p1s = p1s*xsecs[i]/factor
    p2s  = p2s*xsecs[i]/factor
    m2s = m2s*xsecs[i]/factor
    gr_median_narr.SetPoint(i, m, median)
    gr_68_narr.SetPoint(i, m, median)
    gr_68_narr.SetPointError(i, 0, 0, median - m1s, p1s - median)
    gr_95_narr.SetPoint(i, m, median)
    gr_95_narr.SetPointError(i, 0, 0, median - m2s, p2s - median)

gr_theory = TGraph(len(masses))
for i, (m, x) in enumerate(zip(masses, xsecs)):
    gr_theory.SetPoint(i, m, x)

# ======================================================
# Plot
# ======================================================

c = TCanvas("c", "", 700, 600)
c.SetLogy()
gStyle.SetOptStat(0)
gStyle.SetPadBottomMargin(0.12)
gStyle.SetPadLeftMargin(0.14)

xmin, xmax = 1200, 3200
ymin, ymax = 1e-1, 1e+2

frame = TH1F("frame", "", 100, xmin, xmax)
frame.GetYaxis().SetRangeUser(ymin, ymax)
frame.GetXaxis().SetTitle("m_{W_{R}} (TeV)")
frame.GetYaxis().SetTitle("#sigma(pp#rightarrowW_{R})[fb]")#B(W_{R}#rightarroweeqq) [fb]")
frame.Draw()

# Styling
# gr_95.SetFillColor(kOrange - 2)
# gr_68.SetFillColor(kGreen + 1)
gr_median.SetLineColor(kBlue)
gr_median.SetLineStyle(7)
gr_median.SetLineWidth(2)

# gr_95_3jet.SetFillColor(kOrange - 7)
# gr_68_3jet.SetFillColor(kGreen + 3)
gr_median_3jet.SetLineColor(kOrange)
gr_median_3jet.SetLineStyle(2)
gr_median_3jet.SetLineWidth(2)

gr_median_narr.SetLineColor(kGreen)
gr_median_narr.SetLineStyle(2)
gr_median_narr.SetLineWidth(2)

gr_theory.SetLineColor(kRed)
gr_theory.SetLineWidth(2)

# gr_95.Draw("3 same")
# gr_68.Draw("3 same")
gr_median.Draw("L same")

# gr_95_3jet.Draw("3 same")
# gr_68_3jet.Draw("3 same")
gr_median_3jet.Draw("L same")

gr_median_narr.Draw("L same")

# gr_theory.Draw("L same")

# Legend
leg = TLegend(0.50, 0.70, 0.88, 0.88)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.AddEntry(gr_median, "Expected limit (2-jet)", "l")
# leg.AddEntry(gr_68, "68% expected (2-jet)", "f")
# leg.AddEntry(gr_95, "95% expected (2-jet)", "f")
leg.AddEntry(gr_median_3jet, "Expected limit (Wide Signal)", "l")#"Expected limit (3-jet)", "l")
# leg.AddEntry(gr_68_3jet, "68% expected (3-jet)", "f")
# leg.AddEntry(gr_95_3jet, "95% expected (3-jet)", "f")
# leg.AddEntry(gr_theory, "Theory (g_{R}=g_{L})", "l")
leg.AddEntry(gr_median_narr, "Expected limit (Narrow Signal)", "l")
leg.Draw()

# CMS label
cms = TLatex()
cms.SetNDC(True)
cms.SetTextFont(62)
cms.SetTextSize(0.05)
cms.DrawLatex(0.15, 0.91, "CMS")

extra = TLatex()
extra.SetNDC(True)
extra.SetTextFont(52)
extra.SetTextSize(0.04)
extra.DrawLatex(0.26, 0.91, "Preliminary")

lumilabel = TLatex()
lumilabel.SetNDC(True)
lumilabel.SetTextFont(42)
lumilabel.SetTextSize(0.04)
lumilabel.DrawLatex(0.65, 0.91, "138 fb^{-1} (13 TeV)")

desc = TLatex()
desc.SetNDC(True)
desc.SetTextFont(42)
desc.SetTextSize(0.035)
desc.DrawLatex(0.15, 0.80, f"m_{{N}} = M_WR/2 TeV")
desc.DrawLatex(0.15, 0.75, "Combined ee channel")

c.SaveAs(outfile)
print(f" Plot saved as: {outfile}\n")


# #!/usr/bin/env python3
# from ROOT import *
# import sys, os

# gROOT.SetBatch(True)

# # ======================================================
# # Usage:
# # python plot_limits_simple.py <combine_output.root> <theory_xsec.txt> <output_name.pdf>
# # ======================================================

# if len(sys.argv) < 4:
#     print("Usage: python plot_limits_simple.py <combine_output.root> <theory_xsec.txt> <output_name.pdf>")
#     sys.exit(1)

# combine_file = sys.argv[1]
# theory_file = sys.argv[2]
# outfile = sys.argv[3]

# # ======================================================
# # Read theoretical cross section file
# # Columns: MW MN Xsec Unc_Xsec
# # ======================================================

# masses, xsecs, xsec_unc = [], [], []

# with open(theory_file, 'r') as f:
#     for line in f:
#         if line.strip() and not line.startswith('#'):
#             parts = line.split()
#             if len(parts) >= 4:
#                 MW = float(parts[0])
#                 X = float(parts[2])
#                 dX = float(parts[3])
#                 masses.append(MW)
#                 xsecs.append(X)
#                 xsec_unc.append(dX)

# # ======================================================
# # Read expected limits from Combine output
# # ======================================================

# infile = TFile(combine_file, "READ")
# tree = infile.Get("limit")

# mass_limit, exp_limit = [], []
# for event in tree:
#     # match Combine's convention: quantileExpected
#     if abs(event.quantileExpected - 0.5) < 1e-3:
#         mass_limit.append(int(round(event.mh))) 
#         exp_limit.append(event.limit)

# # Sort by mass
# sorted_pairs = sorted(zip(mass_limit, exp_limit))
# mass_limit = [m for m, _ in sorted_pairs]
# exp_limit = [l for _, l in sorted_pairs]

# # ======================================================
# # Create TGraphs
# # ======================================================

# # Theory central
# gr_theory = TGraph(len(masses))
# for i, (m, x) in enumerate(zip(masses, xsecs)):
#     gr_theory.SetPoint(i, m, x)

# # Theory ±1σ band
# gr_theory_band = TGraphAsymmErrors(len(masses))
# for i, (m, x, dx) in enumerate(zip(masses, xsecs, xsec_unc)):
#     gr_theory_band.SetPoint(i, m, x)
#     gr_theory_band.SetPointError(i, 0, 0, dx, dx)

# # Expected limit (already in pb)
# gr_limit = TGraph(len(mass_limit))
# for i, (m, l) in enumerate(zip(mass_limit, exp_limit)):
#     gr_limit.SetPoint(i, m, l)

# # ======================================================
# # Plotting
# # ======================================================

# c = TCanvas("c", "", 700, 600)
# c.SetLogy()
# #c.SetGrid()
# gStyle.SetOptStat(0)
# frame = TH1F("frame", "", 100, min(masses)*0.9, max(masses)*1.1)
# frame.GetYaxis().SetRangeUser(min(xsecs)*0.1, max(xsecs)*10)
# frame.GetXaxis().SetTitle("M_{W} [GeV]")
# frame.GetYaxis().SetTitle("Cross section [pb]")
# frame.Draw()

# # Style
# gr_theory_band.SetFillColorAlpha(kGray+1, 0.4)
# gr_theory_band.SetLineColor(kGray+2)

# gr_theory.SetLineColor(kRed)
# gr_theory.SetLineWidth(2)
# gr_theory.SetMarkerStyle(20)
# gr_theory.SetMarkerColor(kRed)

# gr_limit.SetLineColor(kBlue+2)
# gr_limit.SetLineWidth(2)
# gr_limit.SetLineStyle(2)

# # Draw
# gr_theory_band.Draw("3 same")
# gr_theory.Draw("L same")
# gr_limit.Draw("L same")

# # Legend
# leg = TLegend(0.55, 0.7, 0.88, 0.88)
# leg.SetBorderSize(0)
# leg.SetFillStyle(0)
# leg.AddEntry(gr_theory, "Theoretical cross section", "l")
# leg.AddEntry(gr_theory_band, "#pm 1#sigma Theory uncertainty", "f")
# leg.AddEntry(gr_limit, "Expected limits", "l")
# leg.Draw()

# c.SaveAs(outfile)

# print(f"\n Plot saved as: {outfile}\n")
