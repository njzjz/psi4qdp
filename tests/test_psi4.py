import unittest

import dpdata
import numpy as np


class TestPsi4Driver(unittest.TestCase):
    """Test Psi4 with a hydrogen ion."""

    @classmethod
    def setUpClass(cls):
        cls.system_1 = dpdata.System(
            data={
                "atom_names": ["H"],
                "atom_numbs": [2],
                "atom_types": np.zeros((2,), dtype=int),
                "coords": np.array(
                    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32
                ),
                "cells": np.zeros((1, 3, 3), dtype=np.float32),
                "orig": np.zeros(3, dtype=np.float32),
                "nopbc": True,
            }
        )
        cls.system_2 = cls.system_1.predict(driver="psi4/qdp")
        cls.places = 6

    def test_energy(self):
        energy = self.system_2["energies"].ravel()[0]
        forces = self.system_2["forces"]
        print(energy, forces)


class TestPsi4Minimizer(unittest.TestCase):
    """Test Psi4 with a hydrogen ion."""

    @classmethod
    def setUpClass(cls):
        cls.system_1 = dpdata.System(
            data={
                "atom_names": ["H"],
                "atom_numbs": [2],
                "atom_types": np.zeros((2,), dtype=int),
                "coords": np.array(
                    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32
                ),
                "cells": np.zeros((1, 3, 3), dtype=np.float32),
                "orig": np.zeros(3, dtype=np.float32),
                "nopbc": True,
            }
        )
        cls.system_2 = cls.system_1.minimize(minimizer="psi4/qdp")
        cls.places = 6

    def test_energy(self):
        energy = self.system_2["energies"].ravel()[0]
        forces = self.system_2["forces"]
        coords = self.system_2["coords"]
        print(energy, forces, coords)
