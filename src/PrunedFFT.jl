module PrunedFFT

using FFTW

export pruned_fft

"""
    pruned_fft(x::Matrix{Float64}, n_pad::Int, k_span::Vector{Int64})

    Compute the FFT of a (n, n) matrix x zero padded to (n\\_pad, n\\_pad)
    at frequencies (k, l), k ∈ k\\_span, l ∈ k\\_span.
"""
function pruned_fft(x::Matrix{Float64}, n_pad::Int, k_span::Vector{Int64})

    n, l = size(x)
    nk = length(k_span)

    @assert l === n
    @assert (0 < minimum(k_span)) &  (maximum(k_span) < n_pad)

    # FFT wrt dim 2 (vertical)
    P = plan_fft(zeros(Float64, n_pad))
    tmp = Array{ComplexF64, 2}(undef, n, nk)
    for i in 1:n
        tmp[i, :] = transpose((P*[x[i,:]; zeros(n_pad - n)])[k_span])
    end

    # FFT wrt dim 1 (horizontal)
    P = plan_fft(zeros(ComplexF64, n_pad))
    p_fft = Array{ComplexF64, 2}(undef, nk, nk)
    for j in 1:nk
        p_fft[:, j] = (P*[tmp[:, j]; zeros(n_pad - n)])[k_span]
    end

    p_fft
end

end # module PrunedFFT
