#!/usr/bin/env python3
from ROOT import *
import sys, os

gROOT.SetBatch(True)

if len(sys.argv) < 5:
    print("Usage: python plot_expected_limits_fixedMW.py <combine_output.root> <theory_xsec.txt> <output_name.pdf> <MW_GeV>")
    sys.exit(1)

combine_file = sys.argv[1]
theory_file = sys.argv[2]
outfile = sys.argv[3]
target_MW = float(sys.argv[4])   # e.g. 1000 GeV

# ======================================================
# Read theoretical cross section file
# Columns: MW MN Xsec(pb)
# ======================================================
MN_vals, xsecs = [], []

with open(theory_file, 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 3:
                MW = float(parts[0])
                MN = float(parts[1])
                X = float(parts[2]) * 1000.0   # pb -> fb
                if abs(MW - target_MW) < 1e-3:
                    MN_vals.append(MN)
                    xsecs.append(X)

if not MN_vals:
    print(f"[ERROR] No entries found in theory file for MW = {target_MW} GeV")
    sys.exit(1)

# ======================================================
# Read expected limits from Combine output
# ======================================================
infile = TFile(combine_file, "READ")
tree = infile.Get("limit")

branches = [b.GetName() for b in tree.GetListOfBranches()]
has_mN = any("N" in b for b in branches)
has_mWR = any("mh" in b for b in branches)

exp_median, exp_m1s, exp_p1s, exp_m2s, exp_p2s = {}, {}, {}, {}, {}

for event in tree:
    mWR = getattr(event, "mh", target_MW)
    if abs(mWR - target_MW) > 1e-3:
        continue

    mN = getattr(event, "mN", None)
    if mN is None:
        continue

    q = event.quantileExpected
    if q < 0:  # skip observed
        continue

    val = event.limit * 1000.0  # convert pb -> fb if needed

    if abs(q - 0.5) < 1e-3:
        exp_median[mN] = val
    elif abs(q - 0.16) < 1e-3:
        exp_m1s[mN] = val
    elif abs(q - 0.84) < 1e-3:
        exp_p1s[mN] = val
    elif abs(q - 0.025) < 1e-3:
        exp_m2s[mN] = val
    elif abs(q - 0.975) < 1e-3:
        exp_p2s[mN] = val

MN_vals_sorted = sorted(list(set(MN_vals)))

# ======================================================
# Build TGraphs
# ======================================================
gr_95 = TGraphAsymmErrors(len(MN_vals_sorted))
gr_68 = TGraphAsymmErrors(len(MN_vals_sorted))
gr_median = TGraph(len(MN_vals_sorted))

for i, mN in enumerate(MN_vals_sorted):
    median = exp_median.get(mN, 0)
    m1s = exp_m1s.get(mN, median)
    p1s = exp_p1s.get(mN, median)
    m2s = exp_m2s.get(mN, median)
    p2s = exp_p2s.get(mN, median)

    gr_median.SetPoint(i, mN, median)
    gr_68.SetPoint(i, mN, median)
    gr_68.SetPointError(i, 0, 0, median - m1s, p1s - median)
    gr_95.SetPoint(i, mN, median)
    gr_95.SetPointError(i, 0, 0, median - m2s, p2s - median)

gr_theory = TGraph(len(MN_vals_sorted))
for i, mN in enumerate(MN_vals_sorted):
    gr_theory.SetPoint(i, mN, xsecs[i])

# ======================================================
# Plotting (same as your code)
# ======================================================
c = TCanvas("c", "", 700, 600)
c.SetLogy()
gStyle.SetOptStat(0)
gStyle.SetPadBottomMargin(0.12)
gStyle.SetPadLeftMargin(0.14)

xmin, xmax = min(MN_vals_sorted)*0.9, max(MN_vals_sorted)*1.05
ymin, ymax = 1e-1, max(xsecs)*2

frame = TH1F("frame", "", 100, xmin, xmax)
frame.GetYaxis().SetRangeUser(ymin, ymax)
frame.GetXaxis().SetTitle("m_{N} (GeV)")
frame.GetYaxis().SetTitle("#sigma(pp#rightarrowW_{R}#rightarrow(N,l)) [fb]")
frame.Draw()

# Styling
gr_95.SetFillColor(kOrange - 2)
gr_68.SetFillColor(kGreen + 1)
gr_median.SetLineColor(kBlack)
gr_median.SetLineStyle(7)
gr_median.SetLineWidth(2)

gr_theory.SetLineColor(kRed)
gr_theory.SetLineWidth(2)

gr_95.Draw("3 same")
gr_68.Draw("3 same")
gr_median.Draw("L same")
gr_theory.Draw("L same")

# Legend
leg = TLegend(0.50, 0.70, 0.88, 0.88)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.AddEntry(gr_median, "Expected limit", "l")
leg.AddEntry(gr_68, "68% expected", "f")
leg.AddEntry(gr_95, "95% expected", "f")
leg.AddEntry(gr_theory, "Theory", "l")
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
desc.DrawLatex(0.15, 0.80, f"m_{{W_R}} = {target_MW} GeV")
desc.DrawLatex(0.15, 0.75, "Combined ee channel")

c.SaveAs(outfile)
print(f"Plot saved as: {outfile}\n")

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
