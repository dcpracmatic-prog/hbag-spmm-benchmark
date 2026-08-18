from __future__ import annotations
import os, subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

HERE = os.path.abspath(os.path.dirname(__file__))
SRC_C = os.path.join(HERE, "src", "spmm.c")
SO    = os.path.join(HERE, "hbag", "libhbag.so")

REQUIRED_SYMBOLS = (
    "spmm_hbag_core",
    "spmm_hbag_core_omp",
    "spmm_hbag_core_omp64",
    "spmm_hbag_core_omp64_adaptive",
)

CFLAGS_FULL     = "-O3 -march=native -mavx2 -mfma -fopenmp -Wall -fPIC -shared"
CFLAGS_FALLBACK = "-O3 -Wall -fPIC -shared"

def _build_so(extra):
    return subprocess.run(
        ["gcc", *extra.split(), SRC_C, "-o", SO, "-lm"],
        capture_output=True, text=True,
    )

def _verify_so():
    nm = subprocess.run(["nm", "-D", "--defined-only", SO],
                        capture_output=True, text=True, check=True)
    syms = set()
    for ln in nm.stdout.splitlines():
        parts = ln.split()
        if len(parts) >= 3 and parts[1] in ("T","W","R"):
            syms.add(parts[2])
    missing = [s for s in REQUIRED_SYMBOLS if s not in syms]
    if missing:
        raise RuntimeError(
            "libhbag.so NO exporta los simbolos requeridos: "
            f"{missing}. Tu toolchain no soporta -fopenmp o -mavx2."
        )

class BuildWithGCC(build_py):
    def run(self):
        os.makedirs(os.path.dirname(SO), exist_ok=True)
        for tag, flags in (("full", CFLAGS_FULL), ("fallback", CFLAGS_FALLBACK)):
            res = _build_so(flags)
            if res.returncode == 0:
                print(f"[setup] gcc ({tag}) ok -> {SO}")
                _verify_so()
                super().run()
                return
            last = (res.stderr.strip().splitlines() or [""])[-1]
            print(f"[setup] gcc ({tag}) FAIL: {last}")
        raise RuntimeError("gcc no pudo compilar src/spmm.c en ninguna variante")

setup(
    name="hbag-spmm-adaptive",
    version="0.3.0",
    packages=find_packages(),
    package_data={"hbag": ["libhbag.so"]},
    exclude_package_data={"hbag": ["__pycache__/*", "*.pyc"]},
    include_package_data=True,
    cmdclass={"build_py": BuildWithGCC},
    install_requires=["numpy", "scipy"],
    python_requires=">=3.9",
    description="Sparse x Dense matrix multiply with adaptive OpenMP scheduling (v0.3.0).",
    long_description="See README.md.",
    long_description_content_type="text/markdown",
    license="Apache-2.0",
)
