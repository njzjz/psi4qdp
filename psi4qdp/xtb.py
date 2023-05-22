import numpy as np
from xtb.ase.calculator import XTB as OldXTB


class XTB(OldXTB):
    """ASE calculator for XTB with net charge."""

    def __init__(self, *args, charge=0, **kwargs):
        self.charge = charge
        super().__init__(*args, **kwargs)

    def _create_api_calculator(self):
        initial_charges = np.zeros(len(self.atoms))
        initial_charges[0] = self.charge
        self.atoms.set_initial_charges(initial_charges)
        return super()._create_api_calculator()
