import subprocess as sp
import unittest

import dpdata
import numpy as np


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        dpdata.System(
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
        ).to_deepmd_hdf5("input.h5")

    def test_single_point(self):
        sp.check_output(
            ["psi4qdp", "input.h5", "output.h5", "high_level.h5", "low_level.h5"]
        )

    def test_opt(self):
        sp.check_output(
            [
                "psi4qdp",
                "input.h5",
                "output.h5",
                "high_level.h5",
                "low_level.h5",
                "--opt",
            ]
        )
