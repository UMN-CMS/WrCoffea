#include<iostream>
#include<iomanip>
#include"TH1.h"
#include"TROOT.h"
#include"TH2.h"
#include"TFile.h"
#include"TDirectory.h"
#include"TF1.h"
#include<string>
#include<vector>
#include"TGraphErrors.h"
#include"TGraph.h"
#include"TLegend.h"
#include"TLatex.h"
#include"TCanvas.h"
#include"TGraph2D.h"

void getExclusion(TString fName){
  TFile *f1=new TFile(fName);
  TFile *fout =new TFile("Excl_"+fName,"recreate");
  TH2D *h2_rvalue=(TH2D*)f1->FindObjectAny("mGlmNLSP_r");
  TH2D *h2_XsecUL=(TH2D*)f1->FindObjectAny("mGlmNLSP_XsecUL");

  TCanvas *c3=new TCanvas("c3","c3",1500,1500);
  // c3->SetBottomMargin(0.12);
  // c3->SetLeftMargin(0.12);
  // c3->SetRightMargin(0.12);
  TGraph2D *gr2d_r = new TGraph2D(h2_rvalue);
  //  TGraph2D *gr2dXsec = new TGraph2D(h2_XsecUL);
  cout<<gr2d_r<<endl;
  TGraph2D *gr2dXsec = new TGraph2D(h2_rvalue);
  //  TObjArray* contours = (TObjArray*) gROOT->GetListOfSpecials()->FindObject("mGlmNLSP_r");
  //  TList* list = (TList*)contours->First();
  gr2dXsec->SetNpx(500);
  gr2dXsec->SetNpy(500);
  // gr2d_r->SetNpx(gr2d_r->GetNpx());
  // gr2d_r->SetNpy(gr2d_r->GetNpy());

  TList *list =gr2d_r->GetContourList(1.0);
  cout<<list->GetSize()<<endl;
  TGraph *gr1d=(TGraph*)list->First();
  int k=0;

  gr1d->SetLineColor(kRed);
  gr1d->SetLineWidth(2);

  gr2dXsec->SetTitle(";m_{#tilde{g}}(GeV);m_{#tilde{#chi}_{1}^{0}}(GeV)");
  gr2dXsec->GetXaxis()->SetTitleSize(0.05);
  gr2dXsec->GetXaxis()->SetLabelSize(0.05);
  gr2dXsec->GetYaxis()->SetTitleSize(0.05);
  gr2dXsec->GetYaxis()->SetLabelSize(0.05);
  //  c3->SetLogz();
  //gr2dXsec->GetXaxis()->SetLabelSize(0.4);
  gr2dXsec->SetMaximum(1.0);
  gr2dXsec->SetMinimum(0.0);
  gr2dXsec->Draw("COLZ");
  gr1d->Draw("C");
  gr1d->SetName("gr1d_xseclimit");
  gr2dXsec->SetTitle(";m_{W_{R}}(GeV);m_{N}(GeV)");
  // gr2dXsec->GetXaxis()->SetTitleSize(0.05);
  // gr2dXsec->GetXaxis()->SetLabelSize(0.05);
  // gr2dXsec->GetYaxis()->SetTitleSize(0.05);
  // gr2dXsec->GetYaxis()->SetLabelSize(0.05);


  TString modelName=f1->GetName();
  if(modelName.Contains("T5qqqqHg")) modelName="T5qqqqHg";
  else if(modelName.Contains("T5ttttZg")) modelName="T5ttttZg";
  else if(modelName.Contains("T5bbbbZg")) modelName="T5bbbbZg";
  // TPaveText *text1=new TPaveText(1450,1900,1700,2100,"NB");
  // text1->AddText(modelName);   text1->Draw();

  // TString binType=f1->GetName();
  // if(binType.Contains("SbinV1")) binType="high NJ bins only";
  // else binType="NJ, NbJ bins";
  // TPaveText *text2=new TPaveText(1450,2100,1700,2400,"NB");
  // text2->AddText(binType);   text2->Draw();

  c3->Update();
   c3->Update();
  c3->SaveAs("Excl_"+fName+"_.png");

  fout->cd();
  gr2dXsec->Write();
  gr1d->Write();
  
}
