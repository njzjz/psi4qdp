import os
import numpy as np
import psi4
from dpdata.driver import Driver, Minimizer
from dpdata.unit import EnergyConversion, ForceConversion, LengthConversion

length_convert = LengthConversion("bohr", "angstrom").value()
energy_convert = EnergyConversion("hartree", "eV").value()
force_convert = ForceConversion("hartree/bohr", "eV/angstrom").value()


@Driver.register("psi4/qdp")
class Psi4Driver(Driver):
    def __init__(
        self,
        method: str = "WB97M-D3BJ/def2-TZVPPD",
        charge: int = 0,
        multiplicity=1,
        **kwargs,
    ):
        self.method = method
        self.charge = charge
        self.multiplicity = multiplicity

    """Driver for psi4."""

    def label(self, data: dict) -> dict:
        """Label the system."""
        psi4.set_memory("8 GB")
        psi4.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        types = np.array(data["atom_names"])[data["atom_types"]]
        buff = [f"{self.charge} {self.multiplicity}"]
        for tt, cc in zip(types, data["coords"][0]):
            buff.append(" ".join([tt] + [str(x) for x in cc]))
        psi4.geometry("\n".join(buff))

        G, wfn = psi4.gradient(self.method, return_wfn=True)
        energy = wfn.energy()
        wfn.gradient().print_out()
        gradient = np.array(G)
        data["energies"] = np.array([energy]) * energy_convert
        data["forces"] = -np.array([gradient]) * force_convert
        return data


@Minimizer.register("psi4/qdp")
class Psi4Minimizer(Minimizer):
    """Minimizer for psi4."""

    def __init__(
        self,
        method: str = "WB97M-D3BJ/def2-TZVPPD",
        charge: int = 0,
        multiplicity=1,
        **kwargs,
    ):
        self.method = method
        self.charge = charge
        self.multiplicity = multiplicity

    def minimize(self, data: dict) -> dict:
        """Label the system."""
        psi4.set_memory("8 GB")
        psi4.set_num_threads(4)
        types = np.array(data["atom_names"])[data["atom_types"]]
        buff = [f"{self.charge} {self.multiplicity}"]
        for tt, cc in zip(types, data["coords"][0]):
            buff.append(" ".join([tt] + [str(x) for x in cc]))
        psi4.geometry("\n".join(buff))

        energy, wfn = psi4.optimize(self.method, return_wfn=True)
        gradient = np.array(wfn.gradient())
        coords = np.array(wfn.molecule().geometry())  # Bohr
        data["coords"] = coords.reshape((1, -1, 3)) * length_convert
        data["energies"] = np.array([energy]) * energy_convert
        data["forces"] = -np.array([gradient]) * force_convert
        return data
