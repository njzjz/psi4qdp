import argparse
import traceback
from typing import Tuple

import ase.calculators.calculator
import dpdata
import psi4
import xtb.interface
from dpdata.driver import Driver

from .xtb import XTB


def calculate_correction(
    input: dpdata.System,
    charge: int = 0,
) -> Tuple[dpdata.LabeledSystem, dpdata.LabeledSystem, dpdata.LabeledSystem]:
    """Calculate the correction for the input system.

    Parameters
    ----------
    input : dpdata.System
        input system
    charge : int, optional
        net charge, by default 0

    Returns
    -------
    dpdata.LabeledSystem
        high-level system
    dpdata.LabeledSystem
        low-level system
    dpdata.LabeledSystem
        corrected system
    """
    ll_driver = Driver.get_driver("ase")(XTB(charge=charge, method="GFN2-xTB"))
    ll_sys = input.predict(driver=ll_driver)
    hl_sys = input.predict(driver="psi4/qdp", charge=charge)
    corr_sys = ll_sys.correction(hl_sys)
    return hl_sys, ll_sys, corr_sys


def single_point(args: argparse.Namespace):
    """Run the single-point calculation.

    Parameters
    ----------
    args : argparse.Namespace
        arguments
    """
    input = dpdata.System(args.input, fmt="deepmd/hdf5")
    try:
        hl_sys, ll_sys, corr_sys = calculate_correction(input, charge=args.charge)
    except (
        xtb.interface.XTBException,
        psi4.PsiException,
        ase.calculators.calculator.CalculationFailed,
        RuntimeError,
        Exception,
    ) as e:
        traceback.print_stack()
        hl_sys = dpdata.LabeledSystem()
        ll_sys = dpdata.LabeledSystem()
        corr_sys = dpdata.LabeledSystem()
    hl_sys.to_deepmd_hdf5(args.high_level)
    ll_sys.to_deepmd_hdf5(args.low_level)
    corr_sys.to_deepmd_hdf5(args.output)
    print(hl_sys.data, ll_sys.data, corr_sys.data)


def minimize(args: argparse.Namespace):
    """Minimize and run the correction calculation.

    Parameters
    ----------
    args : argparse.Namespace
        arguments
    """
    input = dpdata.System(args.input, fmt="deepmd/hdf5")
    try:
        hl_sys = input.minimize(minimizer="psi4/qdp", charge=args.charge)
    except (
        psi4.PsiException,
        RuntimeError,
        Exception,
    ) as e:
        traceback.print_stack()
        dpdata.LabeledSystem().to_deepmd_hdf5(args.high_level)
        dpdata.LabeledSystem().to_deepmd_hdf5(args.low_level)
        dpdata.LabeledSystem().to_deepmd_hdf5(args.output)
        return
    hl_sys.to_deepmd_hdf5(args.high_level)

    try:
        ll_driver = Driver.get_driver("ase")(XTB(charge=args.charge, method="GFN2-xTB"))
        ll_sys = hl_sys.predict(driver=ll_driver)
        corr_sys = ll_sys.correction(hl_sys)
    except (
        xtb.interface.XTBException,
        ase.calculators.calculator.CalculationFailed,
        Exception,
    ) as e:
        traceback.print_stack()
        dpdata.LabeledSystem().to_deepmd_hdf5(args.low_level)
        dpdata.LabeledSystem().to_deepmd_hdf5(args.output)
        return
    ll_sys.to_deepmd_hdf5(args.low_level)
    corr_sys.to_deepmd_hdf5(args.output)
    print(hl_sys.data, ll_sys.data, corr_sys.data)


def run(args: argparse.Namespace):
    """Run the correction calculation.

    Parameters
    ----------
    args : argparse.Namespace
        arguments
    """
    if args.opt:
        minimize(args)
    else:
        single_point(args)


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="input HDF5 file")
    parser.add_argument("output", type=str, help="output HDF5 file")
    parser.add_argument("high_level", type=str, help="output high-level HDF5 file")
    parser.add_argument("low_level", type=str, help="output low-level HDF5 file")
    parser.add_argument("--opt", action="store_true", help="optimize the structure")
    parser.add_argument("--charge", type=int, default=0, help="net charge")

    parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)
