"""One galaxy, for fooling around: the rotation curve and the gas and dark mass heatmaps.

Give the numbers on the command line, or leave any out and it asks for them (return takes the default shown):
  .venv/bin/python sparc/single/single.py
  .venv/bin/python sparc/single/single.py --galaxy NGC2403 --disk 0.5 --bulge 0.7 --power 0 --core-gas 0.75

power 0 fits the power to the galaxy. Results go to sparc/single/result/<galaxy>, emptied first; run.txt there lists the
parameters. The core gas beta and the CGM come from paper_params.py. MOND uses its own per-galaxy fit, from mond_upsilon.txt.
"""
import argparse
import shutil
import sys
from pathlib import Path

SINGLE_DIR = Path(__file__).resolve().parent
SPARC_DIR = SINGLE_DIR.parent
MOND_UPSILON_FILE = SPARC_DIR / "mond_upsilon.txt"  # per-galaxy MOND Upsilon factors, from mond_upsilon_fit.py
sys.path.insert(0, str(SPARC_DIR))

from paper_params import PAPER_PARAMS, model_args  # noqa: E402
from sparc_data import LELLI_UPSILON_BULGE, LELLI_UPSILON_DISK, load_properties  # noqa: E402
from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm  # noqa: E402

NUM_MINUTES = 12  # the model only draws heatmaps at 12 or more


def ask(prompt, value, default, convert=float):
    """value if given on the command line, otherwise asks. Return takes the default."""
    while value is None:
        answer = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        try:
            value = convert(answer) if answer else default
        except ValueError as error:
            print(f"  {error}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--galaxy")
    parser.add_argument("--disk", type=float, help="Upsilon_disk")
    parser.add_argument("--bulge", type=float, help="Upsilon_bulge")
    parser.add_argument("--power", type=float, help="W, 0 fits it")
    parser.add_argument("--core-gas", type=float, help="core gas central density per cc")
    args = parser.parse_args()

    names = {name.upper(): name for name in load_properties()}

    def galaxy_name(text):
        if text.upper() not in names:
            raise ValueError(f"{text} is not a SPARC galaxy")
        return names[text.upper()]

    case = PAPER_PARAMS["cases"]["cgm"]
    galaxy = ask("Galaxy", args.galaxy and galaxy_name(args.galaxy), None, galaxy_name)
    disk = ask("Upsilon_disk", args.disk, LELLI_UPSILON_DISK)
    bulge = ask("Upsilon_bulge", args.bulge, LELLI_UPSILON_BULGE)
    power = ask("Power in W (0 fits it)", args.power, PAPER_PARAMS["power_w"])
    core_gas = ask("Core gas central density per cc", args.core_gas, case["core_gas_central_density"])

    out = SINGLE_DIR / "result" / galaxy
    shutil.rmtree(out, ignore_errors=True)
    run_sparc_dm_brute_cgm(**dict(model_args("cgm"), only_galaxies=[galaxy]), add_core_gas_central_density=core_gas,
                           upsilon_disk=disk, upsilon_bulge=bulge, power_override=power, num_minutes=NUM_MINUTES,
                           mond_upsilon_file=MOND_UPSILON_FILE, output_graphs=True, output_dir_override=str(out))

    # keep the galaxy's own images, drop the global plots
    for png in out.glob("*/*.png"):
        png.rename(out / png.name)
    for path in out.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        elif path.name != "run.txt" and not path.name.startswith(galaxy):
            path.unlink()
    print(f"\nResults in {out}: " + ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
