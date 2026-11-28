# Shared by stochastic_thermalize.jl and (until dimfix) stochastic_smallk.jl: predictions from the frozen star.
# K_a for the mode β_c = ê e^{ik·x_c}: total (bond form) and resolved part.
function Kmode(st::Star, k::SVector{2,Float64}, ê::SVector{2,Float64})
    Δ = [ê * (cis(dot(k, r)) - 1) for r in st.r]
    G = sum(st.w[j] * (Δ[j] * transpose(st.r[j])) for j in eachindex(st.r))
    θ = tr(G);  Ω = (G - transpose(G)) / 2;  Sg = (G + transpose(G)) / 2
    S0 = Sg - θ / 2 * I
    Kb = sum(st.w[j] * sum(abs2, Δ[j] - (Ω + θ / 2 * I) * st.r[j]) for j in eachindex(st.r))
    Ktot = st.V * st.T * st.κs * Kb + 0.5 * st.V * st.T * st.κb * abs2(θ)
    Kres = 0.5 * st.V * st.T * (2st.κs * sum(abs2, S0) + st.κb * abs2(θ))
    return Ktot, Kres
end

