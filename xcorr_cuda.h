#ifndef XCORR_CUDA_H
#define XCORR_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include "cufft.h"
#include <iostream>
#include <cmath>
#include <tuple>
#include <vector>

using namespace std;

vector<vector<float>> xcorr(cufftComplex* primSig, cufftComplex* secondSig, int64_t lenPrimSig, int64_t lenSecondSig);

#endif // XCORR_CUDA_H
