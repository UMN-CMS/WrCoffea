#!/bin/bash

#for ((i=36; i<=79; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-50to120_HT-40to70 --start $i
#done

#for ((i=1; i<=72; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-50to120_HT-70to100 --start $i
#done

#for ((i=81; i<=127; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-50to120_HT-100to400 --start $i
#done

#for ((i=1; i<=23; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-50to120_HT-400to800 --start $i
#done

#for ((i=1; i<=52; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-50to120_HT-800to1500 --start $i
#done

#for ((i=20; i<=68; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-50to120_HT-1500to2500 --start $i
#done

#for ((i=7; i<=52; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-50to120_HT-2500 --start $i
#done

#for ((i=1; i<=36; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-120_HT-40to70 --start $i
#done

#for ((i=1; i<=35; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-120_HT-70to100 --start $i
#done

#for ((i=1; i<=38; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-120_HT-100to400 --start $i
#done

#for ((i=1; i<=34; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-120_HT-400to800 --start $i
#done

#for ((i=1; i<=36; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-120_HT-800to1500 --start $i
#done

#for ((i=1; i<=45; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-120_HT-1500to2500 --start $i
#done

#for ((i=1; i<=61; i+=1))
#do
#    python3 skim_files.py DYto2L-4Jets_MLL-120_HT-2500 --start $i
#done

for ((i=63; i<=65; i+=1))
do
    python3 skim_files.py TTto2L2Nu --start $i
done

#for ((i=1; i<=26; i+=1))
#do
#    python3 skim_files.py TbarWplusto2L2Nu --start $i
#done

#for ((i=1; i<=21; i+=1))
#do
#    python3 skim_files.py TWminusto2L2Nu --start $i
#done
