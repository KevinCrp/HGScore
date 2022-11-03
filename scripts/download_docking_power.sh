#!/bin/bash

echo "-------"
echo "Download Docking Power Decoys CASF-2016"
echo "-------"

DATA_DIR=data
DIR_TAR_GZ=$DATA_DIR/casf_16.tar.gz
EXTRACTED_DIR=CASF-2016/decoys_docking

wget http://pdbbind.org.cn/download/CASF-2016.tar.gz -O $DIR_TAR_GZ

echo "Extraction"
tar -xf $DIR_TAR_GZ $EXTRACTED_DIR

mv $EXTRACTED_DIR $DATA_DIR/.
rmdir CASF-2016
