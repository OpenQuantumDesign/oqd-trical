from trical.light_matter.compiler.rule.rewrite.laser_parameters import SetLaserWavelengthFromTransition, SetLaserIntensityFromPiTime

class LaserParametersPass:
    """
    Pass that applies laser parameter conversion rules to the model.
    """

    def __init__(self, laser_wavelength_operands_list, laser_intensity_operands_list):
        self.laser_wavelength_operands_list = laser_wavelength_operands_list
        self.laser_intensity_operands_list = laser_intensity_operands_list

    def run(self, model):
        # Apply wavelength conversion rules
        set_wavelength_rule = SetLaserWavelengthFromTransition()
        for operands in self.laser_wavelength_operands_list:
            set_wavelength_rule.map_set_laser_wavelength_from_transition(model, operands)
        
        # Apply intensity conversion rules
        set_intensity_rule = SetLaserIntensityFromPiTime()
        for operands in self.laser_intensity_operands_list:
            set_intensity_rule.map_set_laser_intensity_from_pi_time(model, operands)
        
        return model
