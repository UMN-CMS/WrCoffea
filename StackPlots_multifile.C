//root -l 'StackPlots_multifile.C("Results","T5bbbbZgPred","2200")'
#include <cmath>

const int n_pl = 4;
bool logx = false;
//TString legend_text[11] = {"No cuts","skimmed","lep-veto","isotrk-veto","Pho-Pt>20","Njets>=2","Dphi-cut","MET>100","MET>250","ST>300","Pho-pt>100"};
TString legend_text[5] ={"Other backgrounds","Nonprompt","Z(ll)+jets","t#bar{t} + tW","Total uncertainty"};//,"Z(#nu#nu)+#gamma","#gamma + jets"};//t #bar{t} + #gamma","t #bar{t} + jets", "Z(#nu#nu)+ #gamma", "W(l#nu) + #gamma","W(l#nu) + jets","#gamma + jets"};//,"Failed acceptance","1e CR"};// {"(0#mu,0e) SR","(1#mu,0e) CR","pMSSM_MCMC_106_19786","pMSSM_MCMC_473_54451"};//{"No cuts","skimmed","lep-veto","isotrk-veto","Dphi-cut","MET>250","ST>300","Pho-pt>100"};
//TString legend_text[4] = {"(0#mu,0e) SR","(1#mu,0e) CR","pMSSM_MCMC_106_19786","pMSSM_MCMC_473_54451"};
int line_width[12] = {3,3,3,3,3,3,3,3,2,2,2,2};
int line_style[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
// /int line_color[n_pl+1] = {kBlack, kRed, kGreen+2, kBlue, kRed};
// /int line_color[n_pl+1] = {kBlack, kRed, kGreen+2, kBlue, kRed};                                                                               
int line_color[9] = {kBlue,kMagenta,kRed+2,kYellow+2,kBlue+2,kMagenta, kRed+2,9,kCyan+2};//,45,kMagenta,kGray+1,kRed,kBlue+2,kMagenta,kCyan};
int line_color1[9]= {kBlue,kGreen+2,kGray+1,kViolet+2,kGreen-2,kYellow+1,kGray+2,kMagenta,kBlue+2};
int line_color2[9] = {kGreen+2,kBlue,kViolet,kGray,kViolet+2,kGreen-2,kYellow+1,kGray+2,kMagenta};
//int line_color[9] = {kMagenta+2, kGray+2, kRed, kGreen+2, kMagenta, kRed - 3, kBlue + 2 , kCyan + 1 , kGreen + 3 };
vector<int> col={kCyan,kGreen,kYellow,kRed,kYellow,kGreen,kCyan,kCyan,kBlue};//kGreen+2,kBlue,kViolet,kGray,kViolet+2,kGreen-2,kYellow+1,kGray+2,kMagenta,kBlue+2,kMagenta,kCyan};
vector<int> Style={1001,1001,1001,1001,1001,1001};
//int line_color[11] = {kPink+1, kRed, kBlue,kGray+1 , kGreen+2, kMagenta, kYellow + 2 , kCyan+3,  kBlue + 2 ,kRed+2,kGreen + 3 };
void decorate(TH1D*,int,int );
 
void decorate(TH1D* hist,int i, int j){
  //  hist->SetLineColor(col[i]);
  // hist->SetFillColor(col[i]);
  // vector<int> col;
  // vector<int> Style;
  // if(j==1){
  //   col={kViolet,kGray+1,kGreen+2,kViolet+2,kGreen-2,kYellow+1,kGray+2,kMagenta,kBlue+2,kMagenta,kCyan};
  //   Style= {3008,1001,3019,3244};
    
  //  }
  // else if(j==2){
  //    col={kBlue,kGreen+2,kGray+1,kViolet+2,kGreen-2,kYellow+1,kGray+2,kMagenta,kBlue+2,kMagenta,kCyan};
  //    Style={1001,3008,1001,3244};
  // }
  // else if(j==3){
  //   col={kGreen+2,kBlue,kViolet,kGray,kViolet+2,kGreen-2,kYellow+1,kGray+2,kMagenta,kBlue+2,kMagenta,kCyan};
  //   Style={3008,1001,3008,1001};
  // }
  //   if(i!=4)
  //     {
  //       hist->SetFillColor(col[i]);

  //       hist->SetFillStyle(Style[i]);
  //     }
  //   else 
  //     {
  //       hist->SetFillColor(kGray+1);

  //       hist->SetFillStyle(1001);
  //       }
    hist->SetLineWidth(3);

  //  if(i<nBG) {                                                                                                                                 
  //hist->SetFillColor(col[i]);
    // if(i!=0)
    //   {
    // 	//	hist->SetFillColor(col[i]);

    //  	hist->SetFillStyle(Style[i]);
    // //   }
    // // else
    // //   {
    // // 	hist->SetFillColor(kGray+2);

    // // 	hist->SetFillStyle(1001);
    // // 	}
    // hist->SetLineWidth(2);
    //}                                                                                                                                           
}

void setLastBinAsOverFlow(TH1D*);
TH1D* setMyRange(TH1D*,double,double);
TH1D* setMyRange(TH1D *h1,double xLow,double xHigh){
  //call it after setting last bin as overflow                                                                                                                               
  double err=0;
  if(xHigh > 13000) return h1;
  if(xLow < -13000) return h1;

  // h1->Print("all");                                                                                                                                                       
  //  h1->GetXaxis()->SetRangeUser(xLow,xHigh);                                                                                                                              
  int nMax=h1->FindBin(xHigh);
  h1->SetBinContent(nMax,h1->IntegralAndError(nMax,h1->GetNbinsX(),err));
  h1->SetBinError(nMax,err);

  //  //cout<<nMax<<endl;                                                                                                                                                      
  //  h1->GetXaxis()->SetRangeUser(xLow,xHigh);                                                                                                                              
  for(int i=nMax+1;i<=h1->GetNbinsX()+1;i++){
    h1->SetBinContent(i,0);
    h1->SetBinError(i,0);
    //    //cout<<":";                                                                                                                                                         
    //h1->GetXaxis()->SetRangeUser(xLow,xHigh);                                                                                                                              
  }
  h1->GetXaxis()->SetRangeUser(xLow,xHigh);
  //cout<<xLow<<"\t"<<xHigh<<"\t"<<"set range"<<endl;
   return h1;

}

TH1D* DrawOverflow(TH1D*);
TH1D* DrawOverflow(TH1D* h,int xmin, int xrange){
    //function to paint the histogram h with an extra bin for overflows
       // This function paint the histogram h with an extra bin for overflows
   UInt_t nx    = h->GetNbinsX()+1;
   Double_t *xbins= new Double_t[nx+1];
   for (UInt_t i=0;i<nx;i++)
     xbins[i]=h->GetBinLowEdge(i+1);
   xbins[nx]=xbins[nx-1]+h->GetBinWidth(nx);
   char *tempName= new char[strlen(h->GetName())+10];
   sprintf(tempName,"%swtOverFlow",h->GetName());
   h->GetXaxis()->SetLimits(xmin,xrange);
   // Book a temporary histogram having ab extra bin for overflows
   TH1D *htmp = new TH1D(tempName, h->GetTitle(), nx, xbins);
   htmp->GetXaxis()->SetRange(xmin,xrange);
   // Reset the axis labels
   htmp->SetXTitle(h->GetXaxis()->GetTitle());
   htmp->SetYTitle(h->GetYaxis()->GetTitle());
   // Fill the new hitogram including the extra bin for overflows
   for (UInt_t i=1; i<=nx; i++)
     htmp->Fill(htmp->GetBinCenter(i), h->GetBinContent(i));
   // Fill the underflows
   htmp->Fill(h->GetBinLowEdge(1)-1, h->GetBinContent(0));
   // Restore the number of entries
   htmp->SetEntries(h->GetEntries());
   // FillStyle and color
   // htmp->SetFillStyle(h->GetFillStyle());
   // htmp->SetFillColor(h->GetFillColor());
   return htmp;
}
void setLastBinAsOverFlow(TH1D* h_hist){
  double lastBinCt =h_hist->GetBinContent(h_hist->GetNbinsX()),overflCt =h_hist->GetBinContent(h_hist->GetNbinsX()+1);
  double lastBinErr=h_hist->GetBinError(h_hist->GetNbinsX()),  overflErr=h_hist->GetBinError(h_hist->GetNbinsX()+1);

  if(lastBinCt!=0 && overflCt!=0)
    lastBinErr = (lastBinCt+overflCt)* (sqrt( ((lastBinErr/lastBinCt)*(lastBinErr/lastBinCt)) + ((overflErr/overflCt)*(overflErr/overflCt)) ) );

  else if(lastBinCt==0 && overflCt!=0)
    lastBinErr = overflErr;
  else if(lastBinCt!=0 && overflCt==0)
    lastBinErr = lastBinErr;
  else lastBinErr=0;

  lastBinCt = lastBinCt+overflCt;
  h_hist->SetBinContent(h_hist->GetNbinsX(),lastBinCt);
  h_hist->SetBinError(h_hist->GetNbinsX(),lastBinErr);
  //cout<<lastBinCt<<"\t"<<"Last bin values"<<endl;

}

// TH1D* setLastBinAsOverFlow(TH1D* h_hist, int xrange){
//   //     h_hist = setMyRange(h_hist,0,xrange);
//   //  h_hist->GetXaxis()->SetRangeUser(0,xrange);
//   double lastBinCt =h_hist->GetBinContent(h_hist->GetNbinsX()),overflCt =h_hist->GetBinContent(h_hist->GetNbinsX());
//   //  //cout<<h_hist->GetNbinsX()<<"\t"<<lastBinCt<<"\t"<<overflCt<<endl;

//   double lastBinErr=h_hist->GetBinError(h_hist->GetNbinsX()),  overflErr=h_hist->GetBinError(h_hist->GetNbinsX()+1);
//   if(lastBinCt!=0 && overflCt!=0)
//     lastBinErr = (lastBinCt+overflCt)* (sqrt( ((lastBinErr/lastBinCt)*(lastBinErr/lastBinCt)) + ((overflErr/overflCt)*(overflErr/overflCt)) ) );

//   else if(lastBinCt==0 && overflCt!=0)
//     lastBinErr = overflErr;
//   else if(lastBinCt!=0 && overflCt==0)
//     lastBinErr = lastBinErr;
//   else lastBinErr=0;
//   //h_temp->GetXaxis()->SetRangeUser(0,xrange);

//   lastBinCt = lastBinCt+overflCt;
//   //  //cout<<lastBinCt<<endl;
//   TH1D* h_temp = (TH1D*)h_hist->Clone();
//   h_temp->SetBinContent(h_hist->GetNbinsX(),lastBinCt);
//   h_temp->SetBinError(h_hist->GetNbinsX(),lastBinErr);
//   //  h_temp->GetXaxis()->SetRangeUser(0,xrange);

//   // h_hist = setMyRange(h_hist,0,xrange);
//   //
//   return h_temp;
// }


// TH1D* setMyRange(TH1D *h1,double xLow,double xHigh){
//   //call it after setting last bin as overflow                                                                                                    
//   double err=0;
//   if(xHigh > 13000) return h1;
//   if(xLow < -13000) return h1;

//   // h1->Print("all");
//   //  h1->GetXaxis()->SetRangeUser(xLow,xHigh);  
//   int nMax=h1->FindBin(xHigh);
//   h1->SetBinContent(nMax,h1->IntegralAndError(nMax,h1->GetNbinsX(),err));
//   h1->SetBinError(nMax,err);

//   //  //cout<<nMax<<endl;
//   //  h1->GetXaxis()->SetRangeUser(xLow,xHigh);
//   for(int i=nMax+1;i<=h1->GetNbinsX()+1;i++){
//     h1->SetBinContent(i,0);
//     h1->SetBinError(i,0);
//     //    //cout<<":";
//     //h1->GetXaxis()->SetRangeUser(xLow,xHigh); 
//   }
//   return h1;
// }

void generate_1Dplot(vector<TH1D*> hist, vector<TH1D*> hist_ratio, char const *tag_name="",char const *xlabel="",char const *ylabel="", float energy=-1, int rebin=-1,double ymin=0,double ymax=0,int xmin=-1,int xmax=-1, char const *leg_head="",  bool normalize=false, bool log_flag=true, bool DoRebin=false, bool save_canvas=true,  vector<string> legend_texts={"nil"}, char const *legend_title="", TString model=""){  

  //cout<<" inside generate 1D plot "<<"\t"<<legend_title<<"\t"<<endl;
  TCanvas *canvas_n1 =      new TCanvas(tag_name, tag_name,1500,1000);
  canvas_n1->Range(-60.25,-0.625,562.25,0.625);
  canvas_n1->SetFillColor(0);
  canvas_n1->SetBorderMode(0);
  canvas_n1->SetBorderSize(2);
  canvas_n1->SetTopMargin(0.05);
  canvas_n1->SetRightMargin(0.035);
  canvas_n1->SetLeftMargin(0.13);
canvas_n1->SetBottomMargin(0.13);
  auto *pad_1 = new TPad("pad_1","pad_1",0.,0.0,1.,0.32); pad_1->Draw();
  pad_1->SetTopMargin(0.04);
  pad_1->SetBottomMargin(0.33);
  pad_1->SetRightMargin(0.035);
  pad_1->SetLeftMargin(0.13);
  auto *p1 = new TPad("p1","p1",0.,0.32,1.,1.);  p1->Draw();
  p1->SetBottomMargin(0.026);
  p1->SetRightMargin(0.035);
  p1->SetLeftMargin(0.13);
  p1->SetTopMargin(0.1);
    p1->cd();
  THStack *hs_var=new THStack("var_Stack","");
  gStyle->SetOptStat(1111111);
       //   gStyle->SetOptStat(0);
       //double pvt_x_min = 0.6;
  double pvt_x_min = 0.75;
  double pvt_x_max = 0.99;
  double pvt_y_min = 0.9;
  //double pvt_dely = 0.18;
  double pvt_dely = 0.15;
  gStyle->SetOptStat(0);
  gROOT->ForceStyle();
  //gStyle->SetOptFit(0);
  vector<TString> legName;
  //TLegend *legend = new TLegend(0.65,0.95,0.99,0.75);
  //  std::string leg_head_str = ;
  double x = 0.15;
  double y = 0.90;
  TLegend *legend;
  //legend = new TLegend(0.60,0.88,0.98,0.72);  
  legend = new TLegend(0.15,0.67,0.5,0.87);  
  legend->SetTextSize(0.035);
  legend->SetLineColor(kWhite);
  legend->SetNColumns(2);
  char* lhead = new char[100];
 //cout<<"before legend fixing "<<endl; 
  sprintf(lhead,"%s ",leg_head);
 auto  legend1 = new TLegend(0.7,0.67,0.95,0.87);
  legend1->SetTextSize(0.035);
  legend1->SetLineColor(kWhite);
  legend1->SetNColumns(2);

  legend1->SetHeader(lhead);

  legend->SetHeader(legend_title);
  legend->SetLineColor(kWhite);
  //cout<<"after legend fixing "<<endl;
  TLegendEntry* leg_entry[20];
  float x_label_size = 0.045;
  //  double ymin = 100000.0;
  //double ymax = 0.0;
  double xrange = xmax;
  // float energy = energyy;
  vector<TH1D*> hist_list_temp;
  //cout<<" hist.size() = "<<hist.size()<<endl;
  for(int i =0;i<(int)hist.size(); i ++) {
    // if(DoRebin) {
    //  hist.at(i)->Rebin(2);

    // }
    //    hist.at(i)= setLastBinAsOverFlow(hist.at(i),xrange);
     

    //    normalize = true;
    if(normalize) {
      hist.at(i)->Scale(1.0/hist.at(i)->Integral());
      hist.at(i)->GetYaxis()->SetTitle("Normalized");
    }
    else {
      hist.at(i)->GetYaxis()->SetTitle("Event yields");
    }
     hist.at(i)->GetXaxis()->SetTitle(xlabel);
    //   hist.at(i)->GetXaxis()->SetRangeUser(xmin,xrange+4);
     //     //cout<<i<<"\t"<<"oinside loop "<<endl;
    hist.at(i)->SetLineWidth(line_width[i]);
    if(i>3){
    hist.at(i)->SetLineStyle(line_style[i-4]);
  
    hist.at(i)->SetLineColor(line_color[i-4]);
    }
    else
      {
	hist.at(i)->SetLineColor(col[i]);
	hist.at(i)->SetFillColor(col[i]);
      }
    //    //cout<<i<<"\t"<<"oinside loop "<<endl;

    hist.at(i)->SetTitle(" ");
    hist.at(i)->GetXaxis()->SetTitleSize(0.05);
    hist.at(i)->GetXaxis()->SetLabelSize(0.05);
    hist.at(i)->GetXaxis()->SetLabelSize(0.0450);
    hist.at(i)->GetYaxis()->SetTitleSize(0.05);
    hist.at(i)->GetYaxis()->SetLabelSize(0.05);
    hist.at(i)->GetYaxis()->SetTitleOffset(1.1);
    hist.at(i)->GetXaxis()->SetTitleOffset(1.1);
    hist.at(i)->GetYaxis()->SetLabelSize(x_label_size);
    //    hist.at(i)->SetLineColor(line_color[i]);
    hist.at(i)->SetTitle(" ");
    //
    hist.at(i)->GetXaxis()->SetTitleSize(0.05);
    hist.at(i)->GetYaxis()->SetTitleSize(0.06);
    hist.at(i)->GetYaxis()->SetLabelSize(0.06);
    hist.at(i)->GetYaxis()->SetTitleOffset(1.);
     decorate(hist.at(i),i, 0);
    hist.at(i)->SetMarkerSize(0.8);
    hist.at(i)->SetMarkerStyle(20);
    //    hist.at(i)->SetMarkerColor(line_color[i]);
    //new ones
    // hist.at(i)->GetXaxis()->SetTitleSize(0.08);
    // hist.at(i)->GetXaxis()->SetLabelSize(0.06);

    // hist.at(i)->GetYaxis()->SetTitleSize(0.07);
    //hist.at(i)->GetYaxis()->SetLabelSize(0.06);

    // hist.at(i)->GetXaxis()->SetTitleOffset(3);
    // hist.at(i)->GetXaxis()->SetLabelOffset(1.6);

    // hist.at(i)->GetYaxis()->SetTitleOffset(0.9);

    // //decorate(hist.at(i),i, which_Lept);
    // hist.at(i)->GetYaxis()->SetNdivisions(506);
    // hist.at(i)->GetXaxis()->SetTitle(title);
    
    // if(DoRebin) {
    //  hist.at(i)->Rebin(2);
    //   //hist.at(i)->Rebin(1);
    // }

  //     double lastBinCt =hist.at(i)->GetBinContent(hist.at(i)->GetNbinsX()),overflCt =hist.at(i)->GetBinContent(hist.at(i)->GetNbinsX()+1);
  // double lastBinErr=hist.at(i)->GetBinError(hist.at(i)->GetNbinsX()),  overflErr=hist.at(i)->GetBinError(hist.at(i)->GetNbinsX()+1);
  // if(lastBinCt!=0 && overflCt!=0)
  //   lastBinErr = (lastBinCt+overflCt)* (sqrt( ((lastBinErr/lastBinCt)*(lastBinErr/lastBinCt)) + ((overflErr/overflCt)*(overflErr/overflCt)) ) );

  // else if(lastBinCt==0 && overflCt!=0)
  //   lastBinErr = overflErr;
  // else if(lastBinCt!=0 && overflCt==0)
  //   lastBinErr = lastBinErr;
  // else lastBinErr=0;

  // lastBinCt = lastBinCt+overflCt;
  // hist.at(i)->SetBinContent(hist.at(i)->GetNbinsX(),lastBinCt);
  // hist.at(i)->SetBinError(hist.at(i)->GetNbinsX(),lastBinErr);
  //  hist.at(i)->GetXaxis()->SetRange(1, hist.at(i)->GetNbinsX() + 1);
    /* hist.at(i)->GetXaxis()->SetRangeUser(x_min[energy],x_max[energy]); */
    //    hist.at(i)= DrawOverflow(hist.at(i));
    legName.push_back(hist.at(i)->GetName());
    if(i>3 && i!=8 ){
    leg_entry[i] = legend->AddEntry(hist.at(i),legend_texts[i-4].c_str(),"l");
    }
    else if (i==8){
      //      hist.at(i)->SetLineColor(kGray+6);
      leg_entry[i] = legend1->AddEntry(hist.at(i),legend_text[4],"f");
      //       leg_entry[i]->SetTextColor(hist.at(i)->GetLineColor()+1);
    }
    else
      leg_entry[i] = legend1->AddEntry(hist.at(i),legend_text[i],"f"); 
    leg_entry[i]->SetTextColor(hist.at(i)->GetLineColor());
    if(i==0 || i==4)
      leg_entry[i]->SetTextColor(hist.at(i)->GetLineColor()+1);
    if(i==8)
      leg_entry[i]->SetTextColor(kGray+4);
    if(hist.at(i)->GetMaximum() > ymax) ymax = hist.at(i)->GetMaximum();
    if(hist.at(i)->GetMinimum() < ymin) ymin = hist.at(i)->GetMinimum();
    // hist.at(i)= setMyRange(hist.at(i),xmin,xmax+4);
    //setLastBinAsOverFlow(hist.at(i));
    //    hist.at(i)->GetXaxis()->SetRangeUser(xmin,xrange+4);

    

  }
  if(ymin == 0.0) ymin = 1e-3;
  if(ymin<0.0) ymin = 1e-4;
  //  if(ymax<=10) ymax=10;
  for(int i = 0;i<(int)hist.size(); i++) {
    if(!normalize) {
      if(model.Contains("TChiWG") || model.Contains("T6ttZg") || model.Contains("TChiNG") || model.Contains("WGJets") || model.Contains("WlnuJets") || model.Contains("ttbarG")|| model.Contains("ttbarJets") || model.Contains("GJets") || model.Contains("ZnunuGJets") ) hist.at(i)->GetYaxis()->SetRangeUser(0.01,1000*ymax);
      else       hist.at(i)->GetYaxis()->SetRangeUser(0.001,100*ymax);    }
    else
      {  hist.at(i)->GetYaxis()->SetRangeUser(0.00001,5.0);
	//	hist.at(i)->GetXaxis()->SetRangeUser(0,xmax_[i]);
      }
    //    p1->SetGrid();
    // if(i==6)
    //   hist.at(i)->Draw("hist ");
    // else if (i>6)
    //    hist.at(i)->Draw("hist sames ");
    // else
      if(i<=3)
	hs_var->Add(hist.at(i));
    hs_var->SetMinimum(0.01);
    hs_var->SetMaximum(ymax*1000);
    //gPad->SetLogu
    //    //cout<<"i Alps "<<i<<endl;
    // if(i>=0) hist.at(i)->Draw("hist ");
    // else hist.at(i)->Draw("hist sames");
	
  }
//  hs_var->SetMinimum(0.0);
//  hs_var->SetMaximum(ymax+0.5);


  hs_var->Draw("HIST");
//   hs_var->Draw("HIST");
  hs_var->GetXaxis()->SetTitleOffset(1.0);
  gPad->Modified(); gPad->Update();
  hs_var->GetXaxis()->SetTitle(xlabel);
  hs_var->GetXaxis()->SetRangeUser(xmin,xmax+0.01*xmax);
  //  hs_var->GetYaxis()->SetTitleSize(
  hs_var->GetYaxis()->SetTitle("Events");//hs_var->GetYaxis()->SetTitle("Events");
  hs_var->SetTitle(0);
  hs_var->GetYaxis()->SetTitleOffset(1.2);
  hs_var->GetXaxis()->SetTitleSize(00.05);
  hs_var->GetXaxis()->SetLabelSize(0.04);
  hs_var->GetXaxis()->SetLabelOffset(2);
  hs_var->GetYaxis()->SetLabelSize(0.04);
  hs_var->GetYaxis()->SetTitleSize(00.055);
  hs_var->GetYaxis()->SetTitleOffset(1.0);

  for(int j=4;j<hist.size()-1;j++){
    hist.at(j)->Draw("Hist sames");
  }
  hist.at(8)->SetLineColor(kGray+6);
    hist.at(8)->SetMarkerColor(kGray);
    hist.at(8)->SetMarkerSize(0.01);

hist.at(8)->SetFillColor(kBlack);
 hist.at(8)->SetFillStyle(3013);
 hist.at(8)->Draw("e2 same");
  // // hs_var->SetMinimum(0.0);
  // // hs_var->SetMaximum(1.5);
  
  
  // hs_var->Draw("BAR HIST");
  // hs_var->Draw("HIST");
  // if(which_TFbins==1) //default 8 bins                                                                                                                                     
  //   hs_var->GetXaxis()->SetRangeUser(0,10);//xmin,xrange);                                                                                                    
  // else if(which_TFbins==2) // v2 TF bins including photon pT>100 and pT<100                                                                                                
  //   hs_var->GetXaxis()->SetRangeUser(0,18);
  // else if(which_TFbins==3) // v3 TF bins including MET<300 and MET>300                                                                                                     
  //   hs_var->GetXaxis()->SetRangeUser(0,18);
  // hs_var->GetXaxis()->SetTitle(title);
  // hs_var->GetXaxis()->SetTitleOffset(1.0);
  // gPad->Modified(); gPad->Update();
  // hs_var->GetXaxis()->SetTitle(title);
  // //  hs_var->GetYaxis()->SetTitleSize(
  // hs_var->GetYaxis()->SetTitle("#frac{N_{SR/CR}}{N_{SR}+N_{CR}}");//hs_var->GetYaxis()->SetTitle("Events");
  // hs_var->SetTitle(0);
  // hs_var->GetYaxis()->SetTitleOffset(1.2);
  // hs_var->GetXaxis()->SetTitleSize(00.05);
  // hs_var->GetXaxis()->SetLabelSize(0.04);
  // hs_var->GetYaxis()->SetLabelSize(0.04);
  // hs_var->GetYaxis()->SetTitleSize(00.055);
  // hs_var->GetYaxis()->SetTitleOffset(1.0);


  legend->Draw();

  legend1->Draw();


  if(log_flag) {
      gPad->SetLogy();
    }
  // if(logx)
  //   gPad->SetLogx();
  gPad->Update(); 
  TLatex* textOnTop = new TLatex();
  //new
  textOnTop->SetTextSize(0.04);
  //  textOnTop->DrawLatexNDC(0.13,0.965,"CMS #it{#bf{Simulation Preliminary}}");

  char* en_lat = new char[500];
  textOnTop->SetTextSize(0.04);
  float inlumi=energy;
  sprintf(en_lat,"#bf{%0.2f fb^{-1} (13 TeV)}",inlumi);
  textOnTop->DrawLatexNDC(0.8,0.91,en_lat);


  gPad->Modified();
  //cout<<"hist_ratio"<<hist_ratio.size()<<endl;
  for(int j=0; j<hist_ratio.size();j++)
    {
      hist_ratio.at(j)->SetLineWidth(2);
      hist_ratio.at(j)->SetLineStyle(1);
      hist_ratio.at(j)->SetMarkerSize(0.2);
      hist_ratio.at(j)->SetLineColor(line_color[j]);
      hist_ratio.at(j)->SetTitle(" ");
      hist_ratio.at(j)->GetXaxis()->SetTitleSize(0.13);
      hist_ratio.at(j)->GetYaxis()->SetTitle("#frac{S}{#sqrt{S+B}}");
      //    hist_ratio.at(j)->GetXaxis()->SetLabelSize(0.1);
      hist_ratio.at(j)->GetYaxis()->SetRangeUser(-0.001,0.03);
    //hist_ratio.at(j)->GetXaxis()->SetRangeUser(xmin,xmax+4);
    // hist_ratio= setMyRange(hist_ratio,xmin,xmax+6);
    //setLastBinAsOverFlow(hist_ratio);

    // if(which_TFbins==1) //default 8 bins                                                                                                    
    //   hist_ratio.at(j)->GetXaxis()->SetRangeUser(0,10);//xmin,xrange);                                                                                                 
    // else if(which_TFbins==2) // v2 TF bins including photon pT>100 and pT<100
    //   hist_ratio.at(j)->GetXaxis()->SetRangeUser(0,18);
    // else if(which_TFbins==3) // v3 TF bins including MET<300 and MET>300                                                                                        
    //    hist_ratio.at(j)->GetXaxis()->SetRangeUser(0,39);
    
    hist_ratio.at(j)->GetXaxis()->SetLabelSize(0.0450);
    hist_ratio.at(j)->GetYaxis()->SetTitleSize(0.13);
    hist_ratio.at(j)->GetYaxis()->SetLabelSize(0.08);
    hist_ratio.at(j)->GetYaxis()->SetTitleOffset(.4);
    hist_ratio.at(j)->SetMarkerSize(1.0);
    hist_ratio.at(j)->SetMarkerStyle(20);
    hist_ratio.at(j)->SetMarkerColor(line_color[j]);
    hist_ratio.at(j)->GetXaxis()->SetTitle(xlabel);
    hist_ratio.at(j)->GetYaxis()->SetNdivisions(505);
    //new
    hist_ratio.at(j)->GetXaxis()->SetTitleSize(0.05);
    hist_ratio.at(j)->GetXaxis()->SetLabelSize(0.11);
    hist_ratio.at(j)->GetYaxis()->SetTitleSize(0.125);
    hist_ratio.at(j)->GetYaxis()->SetNdivisions(505);

    hist_ratio.at(j)->GetXaxis()->SetTitleOffset(1);
    hist_ratio.at(j)->GetYaxis()->SetTitleOffset(0.41);
    hist_ratio.at(j)->GetXaxis()->SetTitleSize(0.14);

    hist_ratio.at(j)->GetYaxis()->SetLabelSize(0.11);
    
    if(normalize){
      hist_ratio.at(j)->GetYaxis()->SetRangeUser(0,5);
      hist_ratio.at(j)->GetXaxis()->SetLabelOffset(0);
      //      hist_ratio.at(j)->GetXaxis()->SetLabelSize(0.4);
    }
    else {
      hist_ratio.at(j)->GetYaxis()->SetRangeUser(0,30);
    }
    // gStyle->SetLabelOffset(1.2);
    // gStyle->SetLabelSize(1.2);
     //     hist_ratio.at(j)->GetXaxis()->SetLabelOffset();
     //hist_ratio.at(j)->GetYaxis()->SetLabelSize(x_label_size);
   pad_1->cd();
   //   pad_1->SetGrid();
   // if(which_TFbins==1){
   // TLine *l =new TLine(0,1.0,10,1.0);   
   // hist_ratio.at(j)->Draw("");
   // l->Draw("sames");
   // TLine *l1 =new TLine(0,1.5,10,1.5);
   // l1->SetLineStyle(7);
   // l1->Draw("sames");
   // TLine *l2 =new TLine(0,0.5,10,0.5);
   // l2->SetLineStyle(7);

   // l2->Draw("sames");
   // }

   // else{
     
   //    TLine *l =new TLine(0,1.0,18,1.0);
   // hist_ratio.at(j)->Draw("");
   // l->Draw("sames");
   // TLine *l1 =new TLine(0,1.5,18,1.5);
   // l1->SetLineStyle(7);
   // l1->Draw("sames");
   // TLine *l2 =new TLine(0,0.5,18,0.5);
   // l2->SetLineStyle(7);

   // l2->Draw("sames");
   // }
   // TLine *l =new TLine(xmin,.01,xrange,.01);
   // hist_ratio.at(j)->Draw("");
   // l->Draw("sames");
   // TLine *l1 =new TLine(xmin,0.02,xrange,0.02);
   // l1->SetLineStyle(7);
   // l1->Draw("sames");
   // TLine *l2 =new TLine(xmin,0.03,xrange,0.03);
   // l2->SetLineStyle(7);

   // l2->Draw("sames");
   if(j==0)
     hist_ratio.at(j)->Draw("EP");
   else
     hist_ratio.at(j)->Draw("EPsames");
    }
   if(normalize){
       //       hist_ratio.at(j)->Scale(1.0/hist_ratio.at(j)->Integral());
       TLine *l =new TLine(xmin,1,xrange,1);
       l->Draw("sames");
       TLine *l1 =new TLine(xmin,1.5,xrange,1.5);
       l1->SetLineStyle(7);
       l1->Draw("sames");
       TLine *l2 =new TLine(xmin,0.5,xrange,0.5);
       l2->SetLineStyle(7);
       l2->Draw("sames");
   }

   else{

      TLine *l =new TLine(xmin,5,xrange,5);
       l->Draw("sames");
       TLine *l1 =new TLine(xmin,10,xrange,10);
       l1->SetLineStyle(7);
       l1->Draw("sames");
       TLine *l2 =new TLine(xmin,15,xrange,15);
       l2->SetLineStyle(7);
      l2->Draw("sames");

   //   TLine *l =new TLine(0,10,50,10);
   // hist_ratio.at(j)->Draw("");
   // l->Draw("sames");
   // TLine *l1 =new TLine(0,12,50,12);
   // l1->SetLineStyle(7);
   // l1->Draw("sames");
   // TLine *l2 =new TLine(0,5,50,5);
   // l2->SetLineStyle(7);

   // l2->Draw("sames");
   }
  char* canvas_name = new char[1000];
  //c->Print(canvas_name);
  
  if(save_canvas) {
    sprintf(canvas_name,"%s.png",tag_name);//.png",tag_name);//_wnormalize.png",tag_name);
     canvas_n1->SaveAs(canvas_name);   
     sprintf(canvas_name,"%s.pdf",tag_name);
    canvas_n1->SaveAs(canvas_name);
  // sprintf(canvas_name,"%s.root",tag_name);
  //   canvas_n1->SaveAs(canvas_name);
    
  }
  
}
const int nfiles=100,nBG=6;                                                                                                                                                              
TFile *f[nfiles];
TFile *f1[nfiles];


void StackPlots_multifile(string pathname, string model, string gluino_m)
{
  char* hname = new char[200];
  char* hname1 = new char[200];
  char* hname2 = new char[200];
  char* hist_name  = new char[200];
  char* hist_name1 = new char[200];
  char* hist_name2 = new char[200];
  char* hist_name3 = new char[200];
  char* hist_name4 = new char[200];
  char* hist_name5 = new char[200];
  char* hist_name6 = new char[200];
  char* hist_name7 = new char[200];
  char* full_path = new char[2000];
  char* full_path1 = new char[2000];
  char* full_path2 = new char[2000];
  char* path2 = new char[2000];
  char* title= new char[2000];
  //string filetag;//=new char[20000];                                                                                                                                                                   
  char* full_path3 = new char[2000];
  char* full_path4 = new char[2000];
  char* full_path5 = new char[2000];
  char* full_path6 = new char[2000];
  char* full_path7 = new char[2000];
  char* full_path8 = new char[2000];
  char* full_path9 = new char[2000];
  char* full_path10 = new char[2000];
  char* full_path11= new char[2000];
  char *leg_head = new char[200];
  int n=0;
  char *dataset=new char[200];
  char *year =new char[200];
  //  float energyy[2]={};
  int n_files=42;
  char *string_png = new char[200];
  vector<string>baseline1, baseline;
  vector<string> legend_texts, legend_texts_v1;
  //  char *legend_title= new char[2000];
  int which_Lept=1;
  int which_TFBins=1;
  vector<string>  Nlsp_m;
  TString Gluino_m = gluino_m.c_str();
  TString Model = model.c_str();
  vector<string> regions ={"wr_ee_boosted_sr","wr_mumu_boosted_sr","wr_ee_resolved_sr","wr_mumu_resolved_sr"};
  vector<string> varName ={"pt_leading_lepton","pt_subleading_lepton","pt_leading_jet","pt_subleading_jet","mass_dilepton","mass_dijet","mass_threeobject_leadlep","mass_fourobject","pt_leading_AK8Jets","mass_twoobject","pt_twoobject"};
  vector <string> xlabel = {"p_{T}^{lead lep}","p_{T}^{sublead lep}","p_{T}^{lead jet}","p_{T}^{sublead jet}","M_{ll}","M_{jj}","M_{N}","M_{lljj}","p_{T}^{AK8}","M_{lJ}","p_{T}^{lJ}"};
  legend_texts ={"ee Boosted SR","#mu#mu Boosted SR","ee Resolved SR","#mu#mu Resolved SR"};
  vector <string>legend_title;
  vector<string> filetag= {"2018","2017","2016postVFP","2016preVFP","2016","FullRun2"};
  vector<float> energyy={ 59.74,41.53,16.5,19.5,36,137.19};
  vector <int> rebin;
  rebin={2,2,4,4,4,4,4,4,4,4,4,4,4,4};
  vector<double> ymin ={0.1, 0.1,0.1, 0.1,0.1, 0.1,0.1, 0.1,0.1, 0.1,0.1};
  vector<double> ymax={1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000};
  vector<double> xmin ={40,40,0,0,0,0,0,0,0,0,0};
  vector<double> xmax={1000,1000,1000,1000,2500,2000,3000,8000,1000,8000,6000};

  //cout<<"different vector sizes "<<endl;
  //cout<<varName.size()<<"\t"<<baseline.size()<<"\t"<<xlabel.size()<<"\t"<<rebin.size()<<"\t"<<xmax.size()<<"\t"<<xmin.size()<<"\t"<<legend_texts.size()<<endl;
  bool flag=false;
  n_files=7;
  if(Model.Contains("WR") && Gluino_m.Contains("1200")){
      Nlsp_m = {"200","400","600","800"};//"50","100","200","400","600","1000"};
      legend_title={"M_{W} = 1200",};      
      legend_texts_v1 = {"M_{N} = 200 GeV","M_{N} = 400 GeV","M_{N} = 600 GeV","M_{N} = 800 GeV"};
      //cout<<"in correct loop"<<endl;
      f[0] = new TFile("WRAnalyzer_signal_WR1200_N200.root");
      f[1] = new TFile("WRAnalyzer_signal_WR1200_N400.root");
      f[2] = new TFile("WRAnalyzer_signal_WR1200_N600.root");
      f[3] = new TFile("WRAnalyzer_signal_WR1200_N800.root");
      n_files=4;
      f1[1] = new TFile("WRAnalyzer_TT_TW.root");
      f1[0] = new TFile("WRAnalyzer_DYJets.root");
      // f1[1] = new TFile("WRAnalyzer_Nonprompt.root");
      // f1[0] = new TFile("WRAnalyzer_Other.root");
    }

    if(Model.Contains("WR") && Gluino_m.Contains("4000")){
      Nlsp_m = {"100","1300","2900","3500"};//"50","100","200","400","600","1000"};                                                                                            
      legend_title={"M_{W} = 4000",};
      legend_texts_v1 = {"M_{N} = 100 GeV","M_{N} = 1300 GeV","M_{N} = 2900 GeV","M_{N} = 3500 GeV"};
      //cout<<"in correct loop"<<endl;
      f[0] = new TFile("WRAnalyzer_signal_WR4000_N100.root");
      f[1] = new TFile("WRAnalyzer_signal_WR4000_N1300.root");
      f[2] = new TFile("WRAnalyzer_signal_WR4000_N2900.root");
      f[3] = new TFile("WRAnalyzer_signal_WR4000_N3500.root");
      n_files=4;
      f1[3] = new TFile("WRAnalyzer_TT_TW.root");
      f1[2] = new TFile("WRAnalyzer_DYJets.root");
      f1[1] = new TFile("WRAnalyzer_Nonprompt.root");
      f1[0] = new TFile("WRAnalyzer_Other.root");
    }

     if(Model.Contains("WR") && Gluino_m.Contains("6000")){
      Nlsp_m = {"100","2100","3500","5700"};//"50","100","200","400","600","1000"};
      legend_title={"M_{W} = 6000",};
      legend_texts_v1 = {"M_{N} = 100 GeV","M_{N} = 2100 GeV","M_{N} = 3500 GeV","M_{N} = 5700 GeV"};
      //cout<<"in correct loop"<<endl;
      f[0] = new TFile("WRAnalyzer_signal_WR6000_N100.root");
      f[1] = new TFile("WRAnalyzer_signal_WR6000_N2100.root");
      f[2] = new TFile("WRAnalyzer_signal_WR6000_N3500.root");
      f[3] = new TFile("WRAnalyzer_signal_WR6000_N5700.root");
      n_files=4;
      f1[3] = new TFile("WRAnalyzer_TT_TW.root");
      f1[2] = new TFile("WRAnalyzer_DYJets.root");
      f1[1] = new TFile("WRAnalyzer_Nonprompt.root");
      f1[0] = new TFile("WRAnalyzer_Other.root");
    }

     if(Model.Contains("WR") && Gluino_m.Contains("8000")){
      Nlsp_m = {"100","2500","4500","7500"};//"50","100","200","400","600","1000"};                                                                                          
      legend_title={"M_{W} = 8000",};
      legend_texts_v1 = {"M_{N} = 100 GeV","M_{N} = 2500 GeV","M_{N} = 4500 GeV","M_{N} = 7500 GeV"};
      //cout<<"in correct loop"<<endl;
      f[0] = new TFile("WRAnalyzer_signal_WR8000_N100.root");
      f[1] = new TFile("WRAnalyzer_signal_WR8000_N2500.root");
      f[2] = new TFile("WRAnalyzer_signal_WR8000_N4500.root");
      f[3] = new TFile("WRAnalyzer_signal_WR8000_N7500.root");
      n_files=4;
      f1[3] = new TFile("WRAnalyzer_TT_TW.root");
      f1[2] = new TFile("WRAnalyzer_DYJets.root");
      f1[1] = new TFile("WRAnalyzer_Nonprompt.root");
      f1[0] = new TFile("WRAnalyzer_Other.root");
    }

     if(Model.Contains("NR") && Gluino_m.Contains("100")){
       Nlsp_m = {"2000","4000","6000","8000"};//"50","100","200","400","600","1000"};
       legend_title={"M_{N} = 100",};
      legend_texts_v1 = {"M_{W} = 2000 GeV","M_{W} = 4000 GeV","M_{W} = 6000 GeV","M_{W} = 8000 GeV"};
      //cout<<"in correct loop"<<endl;
      f[0] = new TFile("WRAnalyzer_signal_WR2000_N100.root");
      f[1] = new TFile("WRAnalyzer_signal_WR4000_N100.root");
      f[2] = new TFile("WRAnalyzer_signal_WR6000_N100.root");
      f[3] = new TFile("WRAnalyzer_signal_WR8000_N100.root");
      n_files=4;
      f1[3] = new TFile("WRAnalyzer_TT_TW.root");
      f1[2] = new TFile("WRAnalyzer_DYJets.root");
      f1[1] = new TFile("WRAnalyzer_Nonprompt.root");
      f1[0] = new TFile("WRAnalyzer_Other.root");
    }

     if(Model.Contains("NR") && Gluino_m.Contains("700")){
       Nlsp_m = {"2000","4000","6000","8000"};//"50","700","200","400","600","7000"};                                                                                       
       legend_title={"M_{N} = 700",};
      legend_texts_v1 = {"M_{W} = 2000 GeV","M_{W} = 4000 GeV","M_{W} = 6000 GeV","M_{W} = 8000 GeV"};
      //cout<<"in correct loop"<<endl;
      f[0] = new TFile("WRAnalyzer_signal_WR2000_N700.root");
      f[1] = new TFile("WRAnalyzer_signal_WR4000_N700.root");
      f[2] = new TFile("WRAnalyzer_signal_WR6000_N700.root");
      f[3] = new TFile("WRAnalyzer_signal_WR8000_N700.root");
      n_files=4;
      f1[3] = new TFile("WRAnalyzer_TT_TW.root");
      f1[2] = new TFile("WRAnalyzer_DYJets.root");
      f1[1] = new TFile("WRAnalyzer_Nonprompt.root");
      f1[0] = new TFile("WRAnalyzer_Other.root");
    }

      if(Model.Contains("NR") && Gluino_m.Contains("1100")){
       Nlsp_m = {"2000","4000","6000","8000"};//"50","1100","200","400","600","11000"};                                                                                        
       legend_title={"M_{N} = 1100",};
      legend_texts_v1 = {"M_{W} = 2000 GeV","M_{W} = 4000 GeV","M_{W} = 6000 GeV","M_{W} = 8000 GeV"};
      //cout<<"in correct loop"<<endl;
      f[0] = new TFile("WRAnalyzer_signal_WR2000_N1100.root");
      f[1] = new TFile("WRAnalyzer_signal_WR4000_N1100.root");
      f[2] = new TFile("WRAnalyzer_signal_WR6000_N1100.root");
      f[3] = new TFile("WRAnalyzer_signal_WR8000_N1100.root");
      n_files=4;
      f1[3] = new TFile("WRAnalyzer_TT_TW.root");
      f1[2] = new TFile("WRAnalyzer_DYJets.root");
      f1[1] = new TFile("WRAnalyzer_Nonprompt.root");
      f1[0] = new TFile("WRAnalyzer_Other.root");
    }

       if(Model.Contains("NR") && Gluino_m.Contains("1100")){
       Nlsp_m = {"2000","4000","6000","8000"};//"50","1100","200","400","600","11000"};                                                                                    
       legend_title={"M_{N} = 1900",};
      legend_texts_v1 = {"M_{W} = 2000 GeV","M_{W} = 4000 GeV","M_{W} = 6000 GeV","M_{W} = 8000 GeV"};
      //cout<<"in correct loop"<<endl;
      f[0] = new TFile("WRAnalyzer_signal_WR2000_N1900.root");
      f[1] = new TFile("WRAnalyzer_signal_WR4000_N1900.root");
      f[2] = new TFile("WRAnalyzer_signal_WR6000_N1900.root");
      f[3] = new TFile("WRAnalyzer_signal_WR8000_N1900.root");
      n_files=4;
      f1[3] = new TFile("WRAnalyzer_TT_TW.root");
      f1[2] = new TFile("WRAnalyzer_DYJets.root");
      f1[1] = new TFile("WRAnalyzer_Nonprompt.root");
      f1[0] = new TFile("WRAnalyzer_Other.root");
    }


    int  i = 4; 
    int  n_var = varName.size();
    int n_cut = baseline.size();
    int n_final = i+n_files;
    int n_start=0;
    for(int i_cut=0; i_cut<regions.size();i_cut++){ //baseline size
      for(int i_var=n_start; i_var<n_var;i_var++)
	{
	  vector<TH1D*> hist_list_Njets;
	  vector<TH1D*> hist_list_Bjets;
	  if ((regions[i_cut].find("boosted") != string::npos)&& ((varName[i_var].find("leading_jet")!=string::npos)|| (varName[i_var].find("_di")!=string::npos) || (varName[i_var].find("threeobject")!=string::npos) || (varName[i_var].find("four")!=string::npos)))
	    {
	      cout<<"skipping "<<varName[i_var].c_str()<<"\t"<<regions[i_cut].c_str()<<endl;	   
	   continue;
	    }
	  if((regions[i_cut].find("resolved") != string::npos) && ((varName[i_var].find("AK8")!=string::npos)||(varName[i_var].find("twoobject")!=string::npos) ))
	    continue;
	  cout<<"keeping"<<"\t"<<varName[i_var].c_str()<<"\t"<<regions[i_cut].c_str()<<endl;
	   for(int i_file=0; i_file<n_final;i_file++)
	     {
	       sprintf(hist_name,"%s_%s",varName[i_var].c_str(),regions[i_cut].c_str());
	       ////cout<<hist_name<<"\t"<<i_cut<<"\t"<<i_var<<"\t"<<i_file<<"\t"<<i_file-6<<endl;
	       ////cout<< "updated "<<hist_name<<"\t"<<regions[i_cut].c_str()<<endl;
	       TDirectory *Td_hist = (TDirectory*)f1[0]->Get(regions[i_cut].c_str());
	       TH1D* h_resp;// = (TH1D*)Td_hist->Get(hist_name);
	       if(i_file<4){
		 Td_hist = (TDirectory*)f1[i_file]->Get(regions[i_cut].c_str());
		 h_resp = (TH1D*)Td_hist->Get(hist_name);
		 cout<<"signal samples"<<"\t"<<h_resp<<"\t"<<i_file<<endl;
	       }
	       else
		 {
		   cout<<"onsdie  "<<hist_name<<"\t"<<i_cut<<"\t"<<i_var<<"\t"<<i_file<<"\t"<<regions[i_cut].c_str()<<"\t"<<f[i_file-4]->GetName()<<endl;
		   Td_hist = (TDirectory*)f[i_file-4]->Get(regions[i_cut].c_str());
		   h_resp = (TH1D*)Td_hist->Get(hist_name);
		 }
	       ////cout<<"resp "<<h_resp->Integral()<<"\t"<<rebin[i_var]<<"\t"<<xmin[i_var]<<"\t"<<xmax[i_var]<<endl;
	       h_resp= setMyRange(h_resp,xmin[i_var],xmax[i_var]+0.01*xmax[i_var]);
	       setLastBinAsOverFlow(h_resp);
	       h_resp->Rebin(rebin[i_var]);
	       if(rebin[i_var]!=1){
                 h_resp->Rebin(2);
               }
	       //h_resp= setMyRange(h_resp,xmin[i_var],xmax[i_var]+0.01*xmax[i_var]);
	       h_resp->Scale(59.74*1000);
	       hist_list_Njets.push_back(h_resp); 
	       double factor=1.0;
	       // h_resp->Scale(factor/h_resp->Integral());
	       // h_resp2->Scale(factor/h_resp2->Integral());
	     }
	   TH1D* h_BGSum ;
	   h_BGSum =(TH1D*)hist_list_Njets.at(0)->Clone();
	   h_BGSum->Add(hist_list_Njets.at(1));
	   h_BGSum->Add(hist_list_Njets.at(2));
	   h_BGSum->Add(hist_list_Njets.at(3));
	   for(int i = 0; i< h_BGSum->GetNbinsX(); i++){
	    double k = h_BGSum->GetBinError(i)/h_BGSum->GetBinContent(i);
	    //	     double l = 0.20*h_BGSum->GetBinContent(i);
	    //////cout<<i <<"\t"<< h_BGSum->GetBinContent(i)<<"\t"<<h_BGSum->GetBinError(i)<<"\t"<<k<<"\t"<<k+0.2<<endl;
	    h_BGSum->SetBinError(i,k*1.2*h_BGSum->GetBinContent(i));	     
	   }
	   hist_list_Njets.push_back(h_BGSum);

	   TH1D* h_SbySB;
	   h_SbySB = (TH1D*)h_BGSum->Clone();
	   for (int n = 4; n<hist_list_Njets.size()-1;n++){
	     TH1D* h_signal_replica = (TH1D*)hist_list_Njets.at(n)->Clone();	     
	      for(int i = 0; i< h_BGSum->GetNbinsX(); i++){
		double B_i = h_BGSum->GetBinContent(i);
		double S1_i = hist_list_Njets.at(n)->GetBinContent(i);
		double Y_i =0;
		double sigmaY=0;
		if(S1_i!=0 && B_i!=0){
		  Y_i = S1_i/sqrt(S1_i+B_i);
		  double varS = hist_list_Njets.at(n)->GetBinError(i)*hist_list_Njets.at(n)->GetBinError(i);
		  double varB = h_BGSum->GetBinError(i)*h_BGSum->GetBinError(i);
		  double dYdS = (0.5*S1_i + B_i) / pow(S1_i+B_i, 1.5);
		  double dYdB = S1_i / (2.0 * pow(S1_i+B_i, 1.5));
		  sigmaY = sqrt(dYdS*dYdS*varS + dYdB*dYdB*varB);
		}
		h_signal_replica->SetBinContent(i, Y_i);
		h_signal_replica->SetBinError(i, sigmaY);		  
	      }
	      setLastBinAsOverFlow(h_signal_replica);
	      h_signal_replica= setMyRange(h_signal_replica,xmin[i_var],xmax[i_var]+0.01*xmax[i_var]);
	      hist_list_Bjets.push_back(h_signal_replica);
	    }
	    ////cout<<" hist_list_Njets.size() "<<hist_list_Njets.size()<<"\t "<<"baseline.size()  "<<baseline.size()<<endl;
	    float energy=energyy[0];
	    int xrange=0.0;
	    
	  sprintf(full_path,"%s/%s_%s_%s_%s_stackedWithbkg",pathname.c_str(),varName[i_var].c_str(),regions[i_cut].c_str(),model.c_str(),gluino_m.c_str());
	  ////cout<<"varName "<< varName[i_var].c_str() <<"\t"<<i_var<<"\t"<<xlabel[i_var].c_str()<<"\t"<<rebin[i_var]<<"\t"<<ymin[i_var]<<"\t"<<ymax[i_var]<<"\t"<<xmin[i_var]<<"\t"<<xmax[i_var]<<"\t"<<legend_title[0].c_str()<<"\t"<<legend_texts_v1.size()<<endl;
	  generate_1Dplot(hist_list_Njets,hist_list_Bjets,full_path,xlabel[i_var].c_str(),"Event yields",energy,rebin[i_var],ymin[i_var],ymax[i_var],xmin[i_var],xmax[i_var],legend_texts[i_cut].c_str(),false,true,false,true,legend_texts_v1,legend_title[0].c_str(), Model);
	
	}

    }
}
      





