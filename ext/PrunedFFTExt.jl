module PrunedFFTExt

using PrunedFFT
using FFTW
using CUDA


"""
    PrunedFFT.pruned_fft(x::CuMatrix{T}, n_pad::Int, k_span::Vector{Int}) where {T <: Real}

    Compute the FFT of a (n, n) matrix x zero padded to (n\\_pad, n\\_pad)
    at frequencies (k, l), k ∈ k\\_span, l ∈ k\\_span.
"""
function PrunedFFT.pruned_fft(x::CuMatrix{T}, n_pad::Int, k_span::Vector{Int}) where {T <: Real}

    n, l = size(x)
    nk = length(k_span)

    @assert l === n
    @assert (0 < minimum(k_span)) &  (maximum(k_span) < n_pad)

    # FFT wrt dim 2 (vertical)
    P = plan_fft(CUDA.zeros(T, n_pad))
    tmp = CuMatrix{ComplexF32}(undef, n, nk)
    for i in 1:n
        @inbounds tmp[i, :] = transpose((P*[x[i,:]; zeros(n_pad - n)])[k_span])
    end

    # FFT wrt dim 1 (horizontal)
    P = plan_fft(CUDA.zeros(ComplexF32, n_pad))
    p_fft = CuMatrix{ComplexF32}(undef, nk, nk)
    for j in 1:nk
        @inbounds p_fft[:, j] = (P*[tmp[:, j]; zeros(n_pad - n)])[k_span]
    end

    p_fft
end

end # module PrunedFFT
