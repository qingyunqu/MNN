#! /bin/zsh
OPENCL="OFF"


cmake .. \
	-DMNN_BUILD_TRAIN=ON \
	-DMNN_OPENCL:BOOL=$OPENCL
	-MNN_USE_OPENCV=ON


make -j8
