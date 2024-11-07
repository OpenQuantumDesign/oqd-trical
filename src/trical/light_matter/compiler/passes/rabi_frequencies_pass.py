from trical.light_matter.compiler.rule.rewrite.get_rabi_frequencies import ComputeMatrixElement, RabiFrequencyFromIntensity, GetRabiFrequenciesDetunings

class RabiFrequenciesPass:
    """
    Pass that applies Rabi frequency calculations and updates the model.
    """

    def run(self, model):
        compute_matrix_element = ComputeMatrixElement()
        for laser_index, laser in enumerate(model.lasers):
            for transition in model.transitions:
                operands = {
                    'laser': laser,
                    'transition': transition,
                }
                matrix_element = compute_matrix_element.map_compute_matrix_element(model, operands)

        # Compute Rabi frequencies from intensities
        rabi_frequency_from_intensity = RabiFrequencyFromIntensity()
        for laser_index, laser in enumerate(model.lasers):
            for transition in model.transitions:
                operands = {
                    'laser_index': laser_index,
                    'intensity': laser.I_0,
                    'transition': transition,
                }
                rabi_frequency = rabi_frequency_from_intensity.map_rabi_frequency_from_intensity(model, operands)

        # Extract Rabi frequencies and detunings from Hamiltonian
        get_rabi_frequencies_detunings_pass = GetRabiFrequenciesDetunings()
        model.H = get_rabi_frequencies_detunings_pass.run(model.H)

        return model
