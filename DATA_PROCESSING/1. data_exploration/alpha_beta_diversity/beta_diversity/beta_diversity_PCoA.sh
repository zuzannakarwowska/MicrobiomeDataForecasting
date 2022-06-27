#!/usr/bin/env bash

#this script:
#   1. reads table
#   2. converts it into biom format
#   3. imports it into QIIME 2 artifact
#   4. calculates Weighted Unifrac and Bray Curtis distance metrics
#   5. plots dist. matrices using PCoA

#!! attention:
#    1. table must be a dataframe with SAMPLES as COLUMNS and OTUs as ROWS
#    2. samples in table must be same as in the metadata file
#    3. we must include phylogenetic tree


TABLE=$1
METADATA=$2
PHYLOGENY=$3

echo ' -------- START --------'

cp $TABLE table.tsv
gsed -i 's/X//g' table.tsv
gsed -i 's/,/\t/g' table.tsv
gsed -i 's/"//g' table.tsv
gsed -i 's/newColName/OTU/g' table.tsv

#read by python
python read_df.py table.tsv
echo 'DATAFRAME READY'
#transform to biom
biom convert -i table.tsv -o converted.biom --to-hdf5

echo 'WELL DONE! table successfully converted to biom file'

#export from biom to qza
qiime tools import \
  --input-path converted.biom \
  --type 'FeatureTable[Frequency]' \
  --input-format BIOMV210Format \
  --output-path table.qza
  
echo 'WELL DONE! biom file successfully imported into QIIME2 artifact'

echo 'CALCULATE BRAY CURTIS'
#calculate Bray-Curtis
qiime diversity beta \
--i-table table.qza \
--p-metric braycurtis \
--o-distance-matrix bc_vector.qza

echo 'WELL DONE! distance matrix successfully calculated'

#calculate distance matrix as input for PCoA
qiime diversity pcoa \
--i-distance-matrix bc_vector.qza \
--o-pcoa bc_distmat_pcoa.qza
echo 'CALCULATE WEIGHTED UNFIRAC'
#make plot
qiime emperor plot \
--i-pcoa bc_distmat_pcoa.qza \
--o-visualization BC_PCoA.qzv \
--m-metadata-file $METADATA
echo 'WELL DONE! PCOA PLOT ON BRAY CURTIS DISTANCE MATRIX WAS CREATED'
echo '                                                                 '
echo 'CALCULATE WEIGHTED UNFIRAC'

#calculate weighted unifrac
qiime diversity beta-phylogenetic \
--i-table table.qza \
--i-phylogeny $PHYLOGENY \
--p-metric unweighted_unifrac \
--o-distance-matrix unifrac_vector.qza

#calculate distance matrix as input for PCoA
qiime diversity pcoa \
--i-distance-matrix unifrac_vector.qza \
--o-pcoa unifrac_distmat_pcoa.qza

#make plot
qiime emperor plot \
--i-pcoa unifrac_distmat_pcoa.qza \
--o-visualization unifrac_plot.qzv \
--m-metadata-file $METADATA

rm table.tsv

echo '------ SCRIPT ENDS ------'

