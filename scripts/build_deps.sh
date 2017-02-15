#!/usr/bin/env bash

BASE_DIR=`pwd`
DEPS_DIR=$BASE_DIR/tmp/deps
DEPS_BUILD_DIR=$BASE_DIR/tmp/deps-build
INSTALL_DIR=$BASE_DIR/tmp/install

mkdir -p $DEPS_BUILD_DIR/TH-build
cd $DEPS_BUILD_DIR/TH-build
cmake -DCMAKE_BUILD_TYPE:STRING=Release \
  -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR \
  $DEPS_DIR/pytorch/torch/lib/TH
make && make install

#cp $BASE_DIR/THNN_CMakeLists.txt $DEPS_DIR/nn/lib/THNN/CMakeLists.txt
mkdir -p $DEPS_BUILD_DIR/THNN-build
cd $DEPS_BUILD_DIR/THNN-build
cmake -DCMAKE_BUILD_TYPE:STRING=Release \
  -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR \
  -DTorch_FOUND="1" \
  -DCMAKE_C_FLAGS:STRING="-I$INSTALL_DIR/include/TH -L$INSTALL_DIR/lib" \
  -DTH_DIR:PATH=$INSTALL_DIR/share/cmake/TH \
  $DEPS_DIR/pytorch/torch/lib/THNN
make && make install

cd $BASE_DIR

