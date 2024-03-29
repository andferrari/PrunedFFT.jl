using PrunedFFT
using Test

using FFTW # to compute f2

n = 128
n_pad = 1024
k_span = collect(512:612)
x = rand(n, n)

f1 = pruned_fft(x, n_pad, k_span)

x_pad = vcat(hcat(x, zeros(n, n_pad - n)), zeros(n_pad - n, n_pad))
f2 = fft(x_pad)[k_span, k_span]

@test sum(abs.(f1 - f2)) ≈ 0.0 atol = 1e-5

# with CUDA

using CUDA

xc = cu(x)
f1c = pruned_fft(xc, n_pad, k_span)

@test sum(abs.(f1c - f2)) ≈ 0.0 atol = 1e-5