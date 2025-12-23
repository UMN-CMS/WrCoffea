import os
import sys
import ROOT
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# --------------------------
# Configuration
# --------------------------
Year = "RunII/2018/RunIISummer20UL18/extremenarrow"
Year1= "RunII/2018/RunIISummer20UL18"
channels = ["EE", "MuMu"]
regions = ["Resolved"]

# Background groups
groups = {
    'DYJets': ['DYJets'],
    'TTbar': ['TTbar'],
    #'nonprompt': ['WJets', 'SingleTop', 'TTbarSemileptonic'],
    #'other': ['TTV', 'Diboson', 'Triboson']
}

# Luminosity uncertainty (lnN)
LumiSyst = "1.20"


WRmasses=[1200,1600,2000,2400,2800,3200]
Nmasses=[[200,400,600,800,1100],[400,600,800,1200,1500],[400,800,1000,1400,1900],[600,800,1200,1800,2300],[600,1000,1400,2000,2700],[800,1200,1600,2400,3000]]

masses=[]

for wrmass in WRmasses:
    for nmass in Nmasses[WRmasses.index(wrmass)]:
        masses.append(f'WR{wrmass}_N{nmass}')

# Mass points
#masses = ['WR2000_N100', 'WR2000_N300', 'WR2000_N500', 'WR2000_N700', 'WR2000_N900', 'WR2000_N1100', 'WR2000_N1300', 'WR2000_N1500', 'WR2000_N1700', 'WR2000_N1900', 'WR4000_N100', 'WR4000_N300', 'WR4000_N500', 'WR4000_N700', 'WR4000_N900', 'WR4000_N1100', 'WR4000_N1300', 'WR4000_N1500', 'WR4000_N1700', 'WR4000_N1900', 'WR4000_N2100', 'WR4000_N2300', 'WR4000_N2500', 'WR4000_N2700', 'WR4000_N2900', 'WR4000_N3100', 'WR4000_N3300', 'WR4000_N3500', 'WR4000_N3700', 'WR4000_N3900', 'WR6000_N100', 'WR6000_N300', 'WR6000_N500', 'WR6000_N700', 'WR6000_N900', 'WR6000_N1100', 'WR6000_N1300', 'WR6000_N1500', 'WR6000_N1700', 'WR6000_N1900', 'WR6000_N2100', 'WR6000_N2300', 'WR6000_N2500', 'WR6000_N2700', 'WR6000_N2900', 'WR6000_N3100', 'WR6000_N3300', 'WR6000_N3500', 'WR6000_N3700', 'WR6000_N3900', 'WR6000_N4100', 'WR6000_N4300', 'WR6000_N4500', 'WR6000_N4700', 'WR6000_N4900', 'WR6000_N5100', 'WR6000_N5300', 'WR6000_N5500', 'WR6000_N5700', 'WR6000_N5900', 'WR8000_N100', 'WR8000_N300', 'WR8000_N500', 'WR8000_N700', 'WR8000_N900', 'WR8000_N1100', 'WR8000_N1300', 'WR8000_N1500', 'WR8000_N1700', 'WR8000_N1900', 'WR8000_N2100', 'WR8000_N2300', 'WR8000_N2500', 'WR8000_N2700', 'WR8000_N2900', 'WR8000_N3100', 'WR8000_N3300', 'WR8000_N3500', 'WR8000_N3700', 'WR8000_N3900', 'WR8000_N4100', 'WR8000_N4300', 'WR8000_N4500', 'WR8000_N4700', 'WR8000_N4900', 'WR8000_N5100', 'WR8000_N5300', 'WR8000_N5500', 'WR8000_N5700', 'WR8000_N5900', 'WR8000_N6100', 'WR8000_N6300', 'WR8000_N6500', 'WR8000_N6700', 'WR8000_N6900', 'WR8000_N7100', 'WR8000_N7300', 'WR8000_N7500', 'WR8000_N7700', 'WR8000_N7900']#[line.strip().replace('WRtoNLtoLLJJ_', '') for line in open("masses.txt")]

# Output folder
outdir = "Datacards"
os.makedirs(outdir, exist_ok=True)

# --------------------------
# Datacard creation loop
# --------------------------
for channel in channels:
    for region in regions:
        hist_type = "mass_fourobject" if region == "Resolved" else "mass_twoobject"
        hist_type1 = "mass_fourobject"
        dir_name = f"wr_{channel.lower()}_{region.lower()}_sr"
        
        for mass in masses:
            card_file = f"{outdir}/WR_narrow_{channel}_{region}_{mass}.txt"
            print(f"[INFO] Creating datacard for {channel} {region} {mass}")
            sig_file = f"WR_Plotter/rootfiles/{Year}/WRAnalyzer_signal_{mass}.root"
            if not os.path.exists(sig_file):
                print(f"[Warning] Missing signal file for {mass}", sig_file)
                continue

            with open(card_file, "w") as f:

                # ---- HEADER ----
                f.write("imax 1\n")
                f.write("jmax *\n")
                f.write("kmax *\n")
                f.write("---------------\n")

                # ---- SHAPES ----
                for group, samples in groups.items():
                    hist_sum = None
                    i = 0
                    for sample in samples:
                        sample_file = f"WR_Plotter/rootfiles/{Year1}/WRAnalyzer_{sample}.root"
                        #print(sample_file)
                        if not os.path.exists(sample_file):
                            print('[Warning] File name that is not found :', sample_file)
                            continue
                        fin = ROOT.TFile.Open(sample_file)
                        h = fin.Get(f"{dir_name}/{hist_type1}_{dir_name}")
                        if not h:
                            print(f"[Warning] Missing hist for {sample} in {region}/{channel}")
                            continue
                        h.SetDirectory(0)
                        h.Scale(138000)
                        if i == 0:
                            hist_sum = h.Clone(f"{group}_{channel}_{region}")
                            hist_sum.SetDirectory(0)
                        else:
                            hist_sum.Add(h)
                        fin.Close()
                        i += 1
                    
                    # if (i >=1):
                    tmpfile = f"WR_Plotter/rootfiles/{Year1}/WRAnalyzer_{group}.root"
                    #     fout = ROOT.TFile(tmpfile, "RECREATE")
                    #     hist_sum.Write(group)
                    #     fout.Close()
                    f.write(f"shapes {group} * {tmpfile} {dir_name}/{hist_type1}_{dir_name} \n")

                # ---- SIGNAL SHAPE ----
                sig_file = f"WR_Plotter/rootfiles/{Year}/WRAnalyzer_signal_{mass}.root"
                if os.path.exists(sig_file):
                    f.write(f"shapes signal * {sig_file} {dir_name}/{hist_type}_{dir_name} $PROCESS_$SYSTEMATIC\n")
                else:
                    print(f"[Warning] Missing signal file for {mass}")
                    continue
                f.write("---------------\n")

                # ---- BINNING ----
                # f.write("bin bin1\n")
                # f.write("observation -1\n")
                f.write("------------------------------\n")

                # ---- PROCESSES ----
                all_bkgs = list(groups.keys())
                all_procs = ["signal"] + all_bkgs
                f.write("bin " + " ".join(["bin1"] * len(all_procs)) + "\n")
                f.write("process " + " ".join(all_procs) + "\n")
                f.write("process " + " ".join([str(i) for i in range(len(all_procs))]) + "\n")
                f.write("rate " + " ".join(["-1"] * len(all_procs)) + "\n")
                f.write("---------------------------------\n")

                # ---- SYSTEMATICS ----
                # Lumi uncertainty
                f.write(f"Lumi lnN {LumiSyst} " + " ".join(["-"] * (len(all_procs) - 1)) + "\n")

                # Shape placeholders
                syst_placeholders = [
                    "JetEn", "JetRes", "ElectronEn", "ElectronRecoSF",
                    "MuonEn", "MuonIDSF", "PU", "Prefire"
                ]
                # for syst in syst_placeholders:
                #     f.write(f"{syst} shapeN2 " + " ".join(["1"] * len(all_procs)) + "\n")

                # Statistical uncertainty
                f.write("scaleAll rateParam * * 138000 [138000,138000]\n")
                f.write("* autoMCStats 0 0 1\n")

            print(f"[✓] Created datacard: {card_file}")