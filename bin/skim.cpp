#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

// ROOT Headers
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TTree.h>
#include <TError.h>
#include <TList.h>

// JSON Parser
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Extract file paths from dataset JSON
std::vector<std::string> extract_file_paths(const std::string& json_path) {
    std::vector<std::string> file_list;
    std::ifstream f(json_path);

    if (!f.is_open()) {
        std::cerr << "Error: Unable to open JSON file: " << json_path << std::endl;
        return file_list;
    }

    json data;
    f >> data;

    for (auto& [dataset_name, dataset_info] : data.items()) {
        if (dataset_info.contains("files") && dataset_info["files"].is_object()) {
            for (auto& [file_path, file_meta] : dataset_info["files"].items()) {
                file_list.push_back(file_path);
            }
        }
    }

    return file_list;
}

void skim_fileset(const std::string& json_path, const std::string& output_file) {
    // Suppress low-severity ROOT info/warning messages during remote reads
    gErrorIgnoreLevel = kError;

    // Enable implicit multi-threading for processing branches in parallel
    ROOT::EnableImplicitMT();

    std::vector<std::string> input_files = extract_file_paths(json_path);
    if (input_files.empty()) {
        std::cerr << "Error: No valid input files extracted from JSON." << std::endl;
        return;
    }

    std::cout << "Loaded " << input_files.size() << " files from JSON. Constructing RDataFrame..." << std::endl;

    // 1. Pass the full vector of file URLs to RDataFrame
    ROOT::RDataFrame df("Events", input_files);

    // 2. Lepton Selection
    // Muons and Electrons passing pT > 45 GeV and |eta| < 2.5
    auto df_leptons = df
        .Define("GoodMuon_pt", "Muon_pt[Muon_pt > 45.0 && abs(Muon_eta) < 2.5]")
        .Define("GoodElectron_pt", "Electron_pt[Electron_pt > 45.0 && abs(Electron_eta) < 2.5]")
        .Define("AllGoodLeptons_pt", [](const ROOT::RVec<float>& m_pt, const ROOT::RVec<float>& e_pt) {
            ROOT::RVec<float> leptons = m_pt;
            leptons.insert(leptons.end(), e_pt.begin(), e_pt.end());
            std::sort(leptons.begin(), leptons.end(), std::greater<float>());
            return leptons;
        }, {"GoodMuon_pt", "GoodElectron_pt"})
        .Define("nGoodLeptons", "AllGoodLeptons_pt.size()")
        .Define("LeadLepton_pt", "nGoodLeptons > 0 ? AllGoodLeptons_pt[0] : 0.0f")
        .Define("SubleadLepton_pt", "nGoodLeptons > 1 ? AllGoodLeptons_pt[1] : 0.0f");

    // 3. Jet Selection
    // AK4: pT > 32 GeV, |eta| < 2.5
    // AK8: pT > 180 GeV, |eta| < 2.5
    auto df_jets = df_leptons
        .Define("nGoodAK4", "Jet_pt[Jet_pt > 32.0 && abs(Jet_eta) < 2.5].size()")
        .Define("nGoodAK8", "FatJet_pt[FatJet_pt > 180.0 && abs(FatJet_eta) < 2.5].size()");

    // 4. Filter Logic
    // (>= 2 leptons passing pT/eta, lead > 52, sublead > 45) AND (>= 2 AK4 jets OR >= 1 AK8 jet)
    auto df_filtered = df_jets.Filter(
        "(nGoodLeptons >= 2 && LeadLepton_pt > 52.0 && SubleadLepton_pt > 45.0) && "
        "(nGoodAK4 >= 2 || nGoodAK8 >= 1)",
        "Skim Selection Cut"
    );

    // 5. Write filtered events to output file
    std::cout << "Executing RDataFrame graph and writing to: " << output_file << std::endl;
    df_filtered.Snapshot("Events", output_file);

    // 6. Merge and write the "Runs" metadata tree across input files
    std::cout << "Copying and merging 'Runs' metadata trees..." << std::endl;
    TList runs_list;
    std::vector<std::unique_ptr<TFile>> open_files;

    for (const auto& file_path : input_files) {
        std::unique_ptr<TFile> fin(TFile::Open(file_path.c_str(), "READ"));
        if (fin && !fin->IsZombie()) {
            TTree* runs_tree = static_cast<TTree*>(fin->Get("Runs"));
            if (runs_tree) {
                runs_list.Add(runs_tree);
                open_files.push_back(std::move(fin)); // Keep TFile alive while merging
            }
        }
    }

    if (!runs_list.IsEmpty()) {
        TFile fout(output_file.c_str(), "UPDATE");
        fout.cd();
        TTree* merged_runs = TTree::MergeTrees(&runs_list);
        if (merged_runs) {
            merged_runs->SetName("Runs");
            merged_runs->Write("", TObject::kOverwrite);
            delete merged_runs;
        }
        fout.Close();
    }

    std::cout << "Done processing fileset!" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string json_path = "RunIII2025_data_fileset.json";
    std::string output_file = "skimmed_output.root";

    if (argc > 1) json_path = argv[1];
    if (argc > 2) output_file = argv[2];

    skim_fileset(json_path, output_file);
    return 0;
}