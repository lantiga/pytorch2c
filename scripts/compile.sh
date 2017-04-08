#!/usr/bin/env bash

if [[ $# -eq 0 ]] ; then
    echo 'Specify filename'
    exit 0
fi

BASE_DIR=`pwd`
DEPS_DIR=$BASE_DIR/tmp/deps
INSTALL_DIR=$BASE_DIR/tmp/install
OUT_BASE_DIR=$BASE_DIR/out
OUT_BUILD_DIR=$BASE_DIR/tmp/out-build

mkdir -p $INSTALL_DIR/include/torch2c
cp $BASE_DIR/include/*.h $INSTALL_DIR/include/torch2c

rm -rf $OUT_BASE_DIR
rm -rf $OUT_BUILD_DIR

FILEPATH=$1
FILENAME=$(basename "$FILEPATH")
FILENAME_NOEXT=${FILENAME%.*}

python $FILEPATH

OUT_DIR=$OUT_BASE_DIR/*
cp $BASE_DIR/scripts/CMakeLists.txt $OUT_DIR
mkdir -p $OUT_BUILD_DIR

cp -R $OUT_DIR/data $OUT_BUILD_DIR
cd $OUT_BUILD_DIR
cmake -DEXECUTABLE=${FILENAME_NOEXT}_test -DSRC=${FILENAME_NOEXT}_test.c -DINSTALL_DIR=$INSTALL_DIR $OUT_DIR && \
make

DYLD_LIBRARY_PATH=$INSTALL_DIR/lib ./${FILENAME_NOEXT}_test

cd $BASE_DIR

