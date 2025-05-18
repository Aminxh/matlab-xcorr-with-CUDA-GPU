#include "cuda_runtime.h"
#include <cufft.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <numeric>
#include <vector>
#include <cuComplex.h>
#include <stdio.h>
#include <cmath>
#include "xcorr_cuda.h"
#include <algorithm>
#include <mkl/mkl.h>
#include <complex>

using namespace std;

#define PI 3.141592653589
#define BLOCKSIZE 256

__global__ void shiftFreqSig_Kernel(cufftComplex* shiftedSig,  cufftComplex* mainSig, int fd, float fs, int64_t sigSize)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < sigSize)
    {
        float angle = 2.0f * PI * fd * (idx + 1.0f)/fs;
        float realPart, imagPart;
        __sincosf(angle, &imagPart, &realPart);

        cufftComplex temp = mainSig[idx];
        cufftComplex calcVal;
        calcVal.x = (temp.x*realPart - temp.y*imagPart);
        calcVal.y = (temp.x*imagPart + temp.y*realPart);

        shiftedSig[idx] = calcVal;
    }
}

__global__ void multipleSigs_conj( cufftComplex* __restrict__ X,  cufftComplex* __restrict__ Y, cufftComplex* __restrict__ R, int sigSize)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < sigSize)
    {
        cufftComplex x = X[idx];
        cufftComplex y = Y[idx];

        cufftComplex multiple;
        multiple.x = x.x * y.x + x.y * y.y;
        multiple.y = - x.x * y.y + x.y * y.x;

        R[idx] = multiple;
    }
}

__global__ void ifftNomalizer_Kernel(cufftComplex* sig, int sigSize, float invVal)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < sigSize)
    {
        sig[idx].x *= invVal;
        sig[idx].y *= invVal;
    }
}

__global__ void fftShift( cufftComplex* sig, cufftComplex* shiftedSig, int sigSize)
{
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < sigSize)
    {
        int halfSize = (sigSize+1) >> 1;
        int shifted_idx = (idx + halfSize) % sigSize;
        shiftedSig[idx] = sig[shifted_idx];
    }
}

__global__ void abs_cuFFT(const cufftComplex* sig, const int64_t* index_range, float* output, int indexRangeSize)
{
    int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < indexRangeSize)
    {
        int idx = index_range[i];
        cufftComplex val = sig[idx];
        output[i] = hypotf(val.x, val.y);
    }

}

//-----------------------------------------------
//
//
// Notice : change local variables in xcorr function yourself
//
//
//----------------------------------------------

 vector<vector<float>> xcorr(cufftComplex* primSig, cufftComplex* secondSig, int64_t lenPrimSig, int64_t lenSecondSig)
{

    auto start = chrono::high_resolution_clock::now();

    // fd initialization ...
    float fs = 500e3;

    // leakage initialization ...
    double leakage_time = 100e-6;
    double leakage_freq = 50;

    // Res_freq_initial initialization ...
    float Res_freq_initial = 1;

    // fd initialization ...
    int fd_lowerBound = -10000;
    int fd_upperBound =  10000;
    int fd_range = ceil((double)((fd_upperBound - fd_lowerBound) + 1)/Res_freq_initial);
    int *fd = new int[fd_range];

    for (int i = 0; i < fd_range ; i++)
        fd[i] = fd_lowerBound + i*Res_freq_initial;

    // approx_delay , lag_center, lag_radius and lags initialization ...
    double approx_delay = 0.003;
    double lag_center = round(approx_delay * fs);
    double lag_radius = abs(lag_center);
    int *lags = new int[2 * (int64_t)lag_radius + 1];

    for (int i = 0; i < 2 * lag_radius + 1; ++i)
        lags[i] = -1 * lag_radius + i;

    // defining Amb and initializating with 0 ...
    vector<vector<float>> Amb(fd_range, vector<float>(2 * lag_radius + 1, 0.0));

    // Performing fft on X in cuda ...
    cufftComplex *d_input, *d_X, *X;
    X = new cufftComplex[lenPrimSig];

    cudaMalloc((void **)&d_input, sizeof(cufftComplex) * lenPrimSig);
    cudaMalloc((void **)&d_X, sizeof(cufftComplex) * lenPrimSig);
    cudaMemcpy(d_input, primSig, sizeof(cufftComplex) * lenPrimSig, cudaMemcpyHostToDevice);

    cufftHandle plan_X;
    cufftPlan1d(&plan_X, lenPrimSig, CUFFT_C2C, 1);
    cufftExecC2C(plan_X, d_input, d_X, CUFFT_FORWARD);

    cudaMemcpy(X, d_X, sizeof(cufftComplex) * lenPrimSig, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cufftDestroy(plan_X);

    // Copy secondSig to GPU to prevent mutiple copying in for loop ...
    cufftComplex *d_secondSig;
    cudaMalloc((void **)&d_secondSig, sizeof(cufftComplex) * lenPrimSig);
    cudaMemcpy(d_secondSig, secondSig, sizeof(cufftComplex) * lenPrimSig, cudaMemcpyHostToDevice);

    // Calculating shifting signal on GPU ...
    cufftComplex *d_shiftedSig;
    cudaMalloc((void **)&d_shiftedSig, sizeof(cufftComplex) * lenPrimSig);

    // Calculating fft of shiftedSig and puting result in d_Y ...
    cufftComplex *d_Y;
    cudaMalloc((void **)&d_Y, sizeof(cufftComplex) * lenPrimSig);

    cufftHandle plan;
    cufftPlan1d(&plan, lenPrimSig, CUFFT_C2C, 1);

    // Calculating X .* Conj(Y) and puting result in d_R
    cufftComplex *d_R;
    cudaMalloc((void **)&d_R, lenPrimSig * sizeof(cufftComplex));

    // Calculatin ifft(X .* Conj(Y)) and puting result to d_final_R ...
    cufftComplex *d_final_R;
    cudaMalloc((void **)&d_final_R, sizeof(cufftComplex) * lenPrimSig);

    cufftHandle ifft_plan;
    cufftPlan1d(&ifft_plan, lenPrimSig , CUFFT_C2C, 1);

    // Defining center, index_range and intializing them ...
    int64_t center = floor(lenPrimSig/2) + 1;
    int64_t *index_range = new int64_t[2 * (int64_t)lag_radius + 1];

    for(int i = 0 ; i < 2 * (int64_t)lag_radius + 1 ; i++)
        index_range[i] = lags[i] + center - 1;

    // Calculating fftShift(d_final_R) and put the result to d_shiftedFFTSig ...
    cufftComplex *d_shiftedFFTSig;
    cudaMalloc((void **)&d_shiftedFFTSig, sizeof(cufftComplex) * lenPrimSig);

    // Copy the index_range to Calculate abs(d_shiftedFFTSig(index_range)) ...
    int64_t *d_index_range;
    float* d_amb_sig;

    cudaMalloc((void **)&d_index_range, (2 * (int64_t)lag_radius + 1) * sizeof(int64_t));
    cudaMalloc((void **)&d_amb_sig, (2 * (int64_t)lag_radius + 1) * sizeof(float));
    cudaMemcpy(d_index_range, index_range, (2 * (int64_t)lag_radius + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);

    for (int i = 0 ; i < fd_range; ++i)
    {

        shiftFreqSig_Kernel<<<(lenPrimSig + BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_shiftedSig, d_secondSig, fd[i], fs, lenPrimSig);

        cufftExecC2C(plan, d_shiftedSig, d_Y, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        multipleSigs_conj<<<(lenPrimSig + BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_X, d_Y, d_R, lenPrimSig);

        cufftExecC2C(ifft_plan, d_R, d_final_R, CUFFT_INVERSE);

        // Normalizing ifft , because ifft in cuda is unnormalized ...
        ifftNomalizer_Kernel<<<(lenPrimSig + BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_final_R, lenPrimSig, 1.0f/lenPrimSig);

        fftShift<<<(lenPrimSig + BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_final_R, d_shiftedFFTSig, lenPrimSig);

        abs_cuFFT<<<(2 * (int64_t)lag_radius + 1 + BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_shiftedFFTSig, d_index_range, d_amb_sig, (2 * (int64_t)lag_radius + 1));

        cudaMemcpy(Amb[i].data(), d_amb_sig, (2 * (int64_t)lag_radius + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    }

    // Free allocated memories on GPU ...
    cudaFree(d_R);
    cudaFree(d_Y);
    cudaFree(d_amb_sig);
    cudaFree(d_final_R);
    cudaFree(d_index_range);
    cudaFree(d_shiftedFFTSig);
    cudaFree(d_shiftedSig);
    cudaFree(d_X);
    cudaFree(d_secondSig);
    cufftDestroy(ifft_plan);
    cufftDestroy(plan);
    delete [] index_range;

    // returning output, you can also adjust this part yourself , like sending by reffrence , pointer and ...
    return Amb;
   }
