from pydantic import ConfigDict

from qutip import Qobj

from oqd_compiler_infrastructure import TypeReflectBaseModel

########################################################################################


class QutipExperiment(TypeReflectBaseModel):
    model_config = ConfigDict(validate_assignments=True, arbitrary_types_allowed=True)

    hamiltonian: Qobj
    duration: float
