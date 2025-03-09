using QuantumToolbox
using NPZ

@time begin
	psi_0 = tensor(
		basis(2, 0),
		basis(2, 0),
		basis(3, 0),
	)

	times = [0.0]
	states = [psi_0]

	H =
		QobjEvo(
			tensor(
				basis(2, 0) * dag(basis(2, 1)),
				eye(2),
				eye(3),
			),
			(p, t) -> 1,
		) +
		QobjEvo(
			tensor(
				basis(2, 1) * dag(basis(2, 0)),
				eye(2),
				eye(3),
			),
			(p, t) -> 1,
		) +
		QobjEvo(
			tensor(
				eye(2),
				basis(2, 0) * dag(basis(2, 1)),
				eye(3),
			),
			(p, t) -> 1,
		) +
		QobjEvo(
			tensor(
				eye(2),
				basis(2, 1) * dag(basis(2, 0)),
				eye(3),
			),
			(p, t) -> 1,
		)

	tspan = LinRange(0, 0.004, 101) .+ times[end]

	sol = sesolve(H, states[end], tspan)

	append!(times, sol.times[2:end])
	append!(states, sol.states[2:end])
end

states = map(s -> s.data, states)
states = transpose(hcat(states...))

npzwrite("__times.npz", times)
npzwrite("__states.npz", states)

rm("__times.npz")
rm("__states.npz")
