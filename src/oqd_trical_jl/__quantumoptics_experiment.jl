using QuantumOptics
using NPZ

@time begin
	psi_0 = tensor(
		basisstate(NLevelBasis(2), 1),
		basisstate(NLevelBasis(2), 1),
		basisstate(FockBasis(2), 1),
	)

	times = [0.0]
	states = [psi_0]

	function H(t, psi)
		1 * tensor(
			transition(NLevelBasis(2), 1, 2),
			identityoperator(NLevelBasis(2)),
			identityoperator(FockBasis(2)),
		) +
		1 * tensor(
			transition(NLevelBasis(2), 2, 1),
			identityoperator(NLevelBasis(2)),
			identityoperator(FockBasis(2)),
		) +
		1 * tensor(
			identityoperator(NLevelBasis(2)),
			transition(NLevelBasis(2), 1, 2),
			identityoperator(FockBasis(2)),
		) +
		1 * tensor(
			identityoperator(NLevelBasis(2)),
			transition(NLevelBasis(2), 2, 1),
			identityoperator(FockBasis(2)),
		)
	end

	tspan = LinRange(0, 0.004, 101) .+ times[end]

	tout, psi_t = timeevolution.schroedinger_dynamic(tspan, states[end], H)

	append!(times, tout[2:end])
	append!(states, psi_t[2:end])
end

states = map(s -> s.data, states)
states = transpose(hcat(states...))

npzwrite("__times.npz", times)
npzwrite("__states.npz", states)

rm("__times.npz")
rm("__states.npz")
