"""
    deim(U, r=size(U, 2))

Select `r` distinct interpolation rows by the discrete empirical interpolation
method. Supports real and complex full-column-rank bases; returns 1-based indices.
"""
function deim(U::AbstractMatrix, r::Integer=size(U, 2))
    1 <= r <= min(size(U)...) || throw(ArgumentError("invalid DEIM rank"))
    all(isfinite, U) || throw(ArgumentError("basis must be finite"))
    p = Int[]
    for j in 1:r
        residual = copy(U[:, j])
        if j > 1
            residual -= U[:, 1:(j - 1)] * (U[p, 1:(j - 1)] \ U[p, j])
        end
        residual[p] .= 0
        i = argmax(abs.(residual))
        precision = eps(real(float(one(eltype(residual)))))
        abs(residual[i]) > 100precision * norm(U[:, j]) ||
            throw(ArgumentError("DEIM basis is numerically rank deficient"))
        push!(p, i)
    end
    return p
end

"""
    gappy_pod(U, oversampling, indices=deim(U))

Append distinct rows using the Gappy-POD smallest-singular-value bound from the
reference algorithm. `oversampling` is the number of additional rows. Uses
absolute squares for complex bases and handles rank-one and tied spectra.
"""
function gappy_pod(U::AbstractMatrix, oversampling::Integer, indices=deim(U))
    n, r = size(U)
    1 <= r <= n || throw(ArgumentError("invalid basis dimensions"))
    all(isfinite, U) || throw(ArgumentError("basis must be finite"))
    p = Int.(collect(indices))
    length(unique(p)) == length(p) || throw(ArgumentError("duplicate sample rows"))
    all(i -> 1 <= i <= n, p) || throw(ArgumentError("sample row out of bounds"))
    length(p) >= r || throw(ArgumentError("at least r initial rows are required"))
    0 <= oversampling <= n - length(p) || throw(ArgumentError("too many extra rows"))
    # Selection is invariant under uniform scaling. Normalize before squaring
    # singular values/row energies so very small or large bases remain usable.
    scale = maximum(abs, U)
    basis = iszero(scale) ? float.(U) : U ./ scale
    for _ in 1:oversampling
        F = svd(basis[p, :])
        # U * V is the complex-safe counterpart of (Vh * U.T).T.
        projected = basis * F.V
        energy = vec(sum(abs2, projected; dims=2))
        g = r == 1 ? 0.0 : max(F.S[end - 1]^2 - F.S[end]^2, 0)
        precision = eps(eltype(F.S))
        if r == 1 || g <= precision * max(F.S[1]^2, precision)
            score = abs2.(projected[:, end])
        else
            z = g .+ energy
            discriminant = max.(z .^ 2 .- 4g .* abs2.(projected[:, end]), 0)
            # Rationalize the subtraction to avoid catastrophic cancellation.
            score = 4g .* abs2.(projected[:, end]) ./ (z .+ sqrt.(discriminant))
        end
        score[p] .= -Inf
        push!(p, argmax(score))
    end
    return p
end
