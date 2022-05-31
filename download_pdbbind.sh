#!/bin/bash

echo "-------"
echo "Download PDBBind version 2020"
echo "-------"

REFINED_TAR_GZ=data/pdbbind_refined_set_2020.tar.gz
GENERAL_TAR_GZ=data/pdbbind_general_set_2020.tar.gz

RAW_DIR=data/raw
REFINED_DIR_NAME=refined-set
GENERAL_DIR_NAME=v2020-other-PL

wget https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_refined.tar.gz -O $REFINED_TAR_GZ
wget https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_other_PL.tar.gz -O $GENERAL_TAR_GZ

echo "Extraction"
tar xzf $REFINED_TAR_GZ
tar xzf $GENERAL_TAR_GZ

rm -rf $REFINED_DIR_NAME/index
rm -rf $REFINED_DIR_NAME/readme
rm -rf $GENERAL_DIR_NAME/index
rm -rf $GENERAL_DIR_NAME/readme

mkdir $RAW_DIR

mv $REFINED_DIR_NAME/* $RAW_DIR/.
mv $GENERAL_DIR_NAME/* $RAW_DIR/.

rmdir $REFINED_DIR_NAME
rmdir $GENERAL_DIR_NAME

echo "Done"
