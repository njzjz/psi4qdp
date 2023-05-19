import argparse
import traceback
from typing import Tuple

import ase.calculators.calculator
import dpdata
import psi4
import xtb.interface
from dpdata.driver import Driver
from xtb.ase.calculator import XTB


def calculate_correction(
    input: dpdata.System,
) -> Tuple[dpdata.LabeledSystem, dpdata.LabeledSystem, dpdata.LabeledSystem]:
    """Calculate the correction for the input system.

    Parameters
    ----------
    input : dpdata.System
        input system

    Returns
    -------
    dpdata.LabeledSystem
        high-level system
    dpdata.LabeledSystem
        low-level system
    dpdata.LabeledSystem
        corrected system
    """
    ll_driver = Driver.get_driver("ase")(XTB(method="GFN2-xTB"))
    ll_sys = input.predict(driver=ll_driver)
    hl_sys = input.predict(driver="psi4/qdp")
    corr_sys = ll_sys.correction(hl_sys)
    return hl_sys, ll_sys, corr_sys


def run(args: argparse.Namespace):
    """Run the correction calculation.

    Parameters
    ----------
    args : argparse.Namespace
        arguments
    """
    input = dpdata.System(args.input, fmt="deepmd/hdf5")
    try:
        hl_sys, ll_sys, corr_sys = calculate_correction(input)
    except (xtb.interface.XTBException, psi4.PsiException, ase.calculators.calculator.CalculationFailed) as e:
        traceback.print_stack()
        hl_sys = dpdata.LabeledSystem()
        ll_sys = dpdata.LabeledSystem()
        corr_sys = dpdata.LabeledSystem()
    hl_sys.to_deepmd_hdf5(args.high_level)
    ll_sys.to_deepmd_hdf5(args.low_level)
    corr_sys.to_deepmd_hdf5(args.output)
    print(hl_sys.data, ll_sys.data, corr_sys.data)


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="input HDF5 file")
    parser.add_argument("output", type=str, help="output HDF5 file")
    parser.add_argument("high_level", type=str, help="output high-level HDF5 file")
    parser.add_argument("low_level", type=str, help="output low-level HDF5 file")

    parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)
