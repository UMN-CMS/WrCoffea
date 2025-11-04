#define PlotLimits_cxx
#include "PlotLimits.h"
#include <TH2.h>
#include<TGraph2DErrors.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <fstream>

using namespace std;

int main(int argc, char* argv[])
{

  if (argc < 2) {
    cerr << "Please give 3 arguments " << "runList " << " " << "outputFileName" << " " << "dataset" << endl;
    return -1;
  }
  const char *inputFileList = argv[1];
  const char *outFileName   = argv[2];
  const char *data          = argv[3];

  PlotLimits ana(inputFileList, outFileName, data);
  cout << "dataset " << data << " " << endl;

  ana.EventLoop(data,inputFileList);

  return 0;
}

void PlotLimits::EventLoop(const char *data,const char *inputFileList) {
  if (fChain == 0) return;

  Long64_t nentries = fChain->GetEntriesFast();
  cout << "nentries " << nentries << endl;
  cout << "Analyzing dataset " << data << " " << endl;

  double minMomMass = 0;
  //   TFile *fxsec=TFile::Open("mStop_Xsecpb_absUnc.root");
  TFile *fxsec=TFile::Open("WR_xsec_map_136TeV.root");//NLO_CrossSection_13TeV_HEPData_Run2Analysis_EXO_20_002.root");
   cout<<" opened root file "<<endl;
   // TH1D *h1_xsec=(TH1D*)fxsec->FindObjectAny("NLO cross sections/Graph2D_y1");
   // cout<<"after getting the 1D file "<<"\t"<<h1_xsec->GetName()<<endl;
   // TH1D *h1_xsec=(TH1D*)fxsec->FindObjectAny("mStopXsec");                                                        
 
   //  TFile *fxsec1=TFile::Open("T6ttZG_Summer16v3_MassScan.root"); minMomMass = 800.0;
  // TH1D *h1_xsec=(TH1D*)fxsec->FindObjectAny("mStopXsec");
   TFile *fmass=TFile::Open("WR_xsec_map_136TeV.root");//NLO_CrossSection_13TeV_HEPData_Run2Analysis_EXO_20_002.root"); //minMomMass = 1800.0; // not used anywhere
   TDirectory *dir = (TDirectory*)fxsec->Get("NLO cross sections");
   if (!dir) {
     cout << "Directory not found!" << endl;
     //return;
   }
   // Now get the histogram/graph inside that directory
   //TGraph2DErrors *h1_xsec = (TGraph2DErrors*)dir->Get("Graph2D_y1");
   TH2F *h1_xsec = (TH2F*)fxsec->Get("WR_xsec_map");
   if (!h1_xsec) {
     cout << "Histogram not found!" << endl;
     return;
   }
   
   cout<<"Taking xsec from "<<fxsec->GetName()<<" and this hist"<<h1_xsec->GetName()<<" "<<h1_xsec->GetTitle()<<endl;

  TString s_data=data;
  // if(s_data.Contains("T6ttZg")){
  //   fxsec=TFile::Open("mStop_Xsecpb_absUnc.root");
  //   h1_xsec=(TH1D*)fxsec->FindObjectAny("mStopXsec");
  //   minMomMass =1000.0;
  // }
    cout<<"Taking xsec from "<<fxsec->GetName()<<" and this hist"<<h1_xsec->GetName()<<" "<<h1_xsec->GetTitle()<<endl;

  Long64_t nbytes = 0, nb = 0;
  int decade = 0;
  int evtSurvived=0;

  vector<TString> fNames;
  ifstream filein(inputFileList);
  if(filein.is_open()){
    string line1;
    while(getline(filein,line1))
      fNames.push_back(line1);    
  }
  else cout<<"Could not open file: "<<inputFileList<<endl;
  filein.close();
  cout<<"Using Asymptotic "<<fNames.size()<<endl;

  for (Long64_t jentry=0; jentry<nentries;jentry++) {
   
    //cout<<" ==============print number of events done == == == == == == == ="<<endl;
    double progress = 10.0 * jentry / (1.0 * nentries);
    int k = int (progress);
    // if (k > decade)
    //   cout << 10 * k << " %" <<endl;
    // decade = k;
    // cout<<"j:"<<jentry<<" fcurrent:"<<fCurrent<<endl;
    // ===============read this entry == == == == == == == == == == == 
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    //    cout<<fCurrent<<" "<<fNames[fCurrent]<<endl;

    //    if(quantileExpected > 0) continue;
    double m1=mh;
    double m2=(mh-m1)*10000;
    double mGl=m1,mNLSP=round(m2),xsec=0,xsecUnc=0;
    int WR_mass = int(mh / 10000);    // integer division
    int N_mass  = int(mh) % 10000;    // remainder
    mGl = WR_mass;
    mNLSP = N_mass;
    //cout<<"j:"<<jentry<<" fcurrent:"<<fCurrent<<"\t"<<mh<<"\t"<<mGl<<"\t"<<mNLSP<<"\t"<<m1<<"\t"<<m2<<endl;
    // int nPoints = h1_xsec->GetN();
    // double *x = h1_xsec->GetX();  // WR masses
    // double *y = h1_xsec->GetY();  // N masses
    // double *z = h1_xsec->GetZ();  // cross sections
    // double *ez = h1_xsec->GetEZ(); // cross section uncertainties
    // cout<<"inside get point "<<endl;
    
    // for (int i = 0; i < nPoints; i++) {
    //   if (fabs(x[i] - WR_mass) < 5 && fabs(y[i] - N_mass) < 5) {
    //     xsec = z[i];
    //     xsecUnc = ez[i];
    //     break;
    //   }
    // }


    // - Run-3 where it is a Th2D and not TGraph Errors
    // int binX = h1_xsec->GetXaxis()->FindBin(mGl);
    // int binY = h1_xsec->GetYaxis()->FindBin(mNLSP);
    // cout<<"j:"<<jentry<<" fcurrent:"<<fCurrent<<endl;
    // xsec = h1_xsec->GetBinContent(binX, binY);
    // xsecUnc = h1_xsec->GetBinError(binX, binY);

    xsec = h1_xsec->GetBinContent(h1_xsec->FindBin(mGl));
    xsecUnc = h1_xsec->GetBinError(h1_xsec->FindBin(mGl));
    cout<< quantileExpected<<"\t"<<mGl<<"\t"<<mNLSP<<"\t"<<limit<<"\t"<<xsec<<"\t"<<xsecUnc<<endl;
     if (xsec < 0) {
      std::cerr << "No cross section found for WR=" << WR_mass
                << ", N=" << N_mass << std::endl;
    } else {
      std::cout << "WR=" << WR_mass << ", N=" << N_mass
                << ", xsec=" << xsec << " +/- " << xsecUnc << std::endl;
    }

    if(quantileExpected < 0){
      h2_mGlmNLSP_r->Fill(mGl,mNLSP,limit);
      h2_mGlmNLSP_XsecUL->Fill(mGl,mNLSP,xsec/limit);
      h2_mGlmNLSP_r_fb->Fill(mGl,mNLSP,limit);
      h2_mGlmNLSP_XsecUL_fb->Fill(mGl,mNLSP,xsec/limit);

      h2_mGlmNLSP_rUnc->Fill(mGl,mNLSP,limitErr);
      h2_mGlmNLSP_r_XsecUp->Fill(mGl,mNLSP,( limit * (xsec/(xsec+xsecUnc)) ));
      h2_mGlmNLSP_r_XsecDn->Fill(mGl,mNLSP,( limit * (xsec/(xsec-xsecUnc)) ));
    }
    if(abs(quantileExpected - 0.1599999) <= 0.0001){
      h2_mGlmNLSP_16pc->Fill(mGl,mNLSP,limit);
    }
    if(abs(quantileExpected - 0.8399999) <= 0.0001){
      h2_mGlmNLSP_84pc->Fill(mGl,mNLSP,limit);
    }
    if(abs(quantileExpected - 0.5) <= 0.0001){
      h2_mGlmNLSP_median->Fill(mGl,mNLSP,limit);
    }

     if(abs(quantileExpected - 0.025) <= 0.0001){
      h2_mGlmNLSP_2p5pc->Fill(mGl,mNLSP,limit);
    }
    if(abs(quantileExpected - 0.975) <= 0.0001){
      h2_mGlmNLSP_97p5pc->Fill(mGl,mNLSP,limit);
    }

    //    cout<<GluinoMass<<" "<<NLSPMass<<endl;
  }
}
