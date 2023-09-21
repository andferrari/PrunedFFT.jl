Compute the FFT of a `(n, n)`matrix `x`, zero padded to `(n_pad, n_pad)`
at frequencies `(k, l)`, `k ∈ k_span`, `l ∈ k_span` with a memory footprint `O(max(k_span)², n_pad)`.

```julia
using PrunedFFT

n = 128
x = rand(n, n)

n_pad = 1024
k_span = collect(512:612)

f1 = pruned_fft(x, n_pad, k_span)
```

If `CUDA.jl` is loaded, computation is on GPU
```julia
using PrunedFFT
using CUDA

n = 128
xc = cu(rand(n, n))

n_pad = 1024
k_span = collect(512:612)

f1 = pruned_fft(xc, n_pad, k_span)
```

