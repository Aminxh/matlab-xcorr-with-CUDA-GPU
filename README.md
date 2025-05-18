# ⚡ CUDA Accelerated Cross-Correlation Engine (1D Ambiguity Function)

> 🚀 Real-time GPU implementation of frequency-compensated cross-correlation via CUDA, adapted from MATLAB:
>
> ```matlab
> xcorr(Prim_sig.' .* exp(-1i * 2 * pi * fd(i) * (1:N) / 500e3), Second_sig.', lag_radius)
> ```

---

## 📌 Overview

This project implements a fast, parallelized version of frequency-drift compensated cross-correlation on CUDA. It is designed to construct a 2D **ambiguity function** surface across frequency and delay axes using matched filtering logic. It is mathematically faithful to the original MATLAB expression above, and is specifically optimized for large input signals.

---

## 🎯 Goals

- Perform Doppler-shifted cross-correlation efficiently using the GPU
- Reuse FFT results across frequency bins
- Match MATLAB results in accuracy
- Keep **time complexity linear: O(N)**

---

## 🧠 MATLAB → CUDA Mapping

| MATLAB Expression | CUDA Equivalent |
|-------------------|------------------|
| `Prim_sig .* exp(-j2πfdt)` | `shiftFreqSig_Kernel` |
| `fft(X)` | `cufftExecC2C(..., CUFFT_FORWARD)` |
| `X .* conj(Y)` | `multipleSigs_conj` |
| `ifft(R)` | `cufftExecC2C(..., CUFFT_INVERSE)` |
| `fftshift(result)` | `fftShift` |
| `abs(result(lag_range))` | `abs_cuFFT` |

---

## 🧰 Key Features

- ⚙️ Frequency shifting in GPU (`exp(-j2πfdt)`)
- ⚡ Full FFT/IFFT using cuFFT
- 🔁 Reuse of FFT of reference signal
- 🧮 Complex multiplication kernel with conjugate logic
- 🎯 Extracted window of lags around approximate delay
- 📈 Time Complexity: **O(N)** per Doppler bin (FFT-based method)

---
## 🧾 Input Parameters
| Parameter	| Description |
|-------------------|------------------|
| `primSig`	| `Primary signal (reference)` |
| `secondSig` | `Signal to correlate against` |
| `fs` | `Sampling frequency (default: 500e3 Hz)`|
| `fd`	| `Doppler bins (default: -10 kHz to +10 kHz)` |
| `lag_radius` | `Samples ± around approximate delay (default: ~1500)` |

---
## 🧪 Performance Benchmark
| System Component	| Value |
|-------------------|------------------|
| `GPU`	| `NVIDIA RTX 3080` |
| `Input Signal Size`	| `500,000 complex samples` |
| `Doppler Search Space` |	`20,001 bins` |
| `Lag Window` |	`±1500 samples` |
| `Execution Time`	| `~18 milliseconds` |
| `Speedup (vs CPU)` |	`~40× faster` |

## 🧠 Notes

    Time complexity is linear per frequency bin.

    FFT of reference signal is precomputed once.

    Frequency resolution and range can be adjusted.

    Designed to work on large signal lengths efficiently.

## 🧪 Test Environment

    OS      : Ubuntu 22.04
    Compiler: nvcc 12.1
    GPU     : NVIDIA RTX 3080
    CUDA    : 11.8

## 📜 License

    MIT License — free to use, adapt, and distribute with attribution.
    
## ✨ Credits
Created with a focus on real-time signal processing acceleration, based on mathematical fidelity to MATLAB's xcorr function.

## 🤝 Contributing

Issues and pull requests are welcome! Feel free to:

    Add support for 2D signals

    Optimize kernel memory reuse

    Port to other GPU libraries (HIP, OpenCL)




