using PackageCompiler

create_sysimage(
	["QuantumOptics", "NPZ"];
	sysimage_path = "QuantumOpticsBackend.so",
	precompile_execution_file = "__quantumoptics_experiment.jl",
)

create_sysimage(
	["QuantumToolbox", "NPZ"];
	sysimage_path = "QuantumToolboxBackend.so",
	precompile_execution_file = "__quantumtoolbox_experiment.jl",
)
