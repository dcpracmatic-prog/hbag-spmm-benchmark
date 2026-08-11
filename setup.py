from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import subprocess
import os

class CustomBuildPy(build_py):
    def run(self):
        # 1. Ejecutar el proceso normal de empaquetado
        super().run()
        
        # 2. Compilar src/spmm.c directamente hacia la carpeta build del paquete hbag
        src_c = os.path.join(os.path.dirname(__file__), 'src', 'spmm.c')
        target_dir = os.path.join(self.build_lib, 'hbag')
        so_path = os.path.join(target_dir, 'libhbag.so')

        os.makedirs(target_dir, exist_ok=True)
        cmd = f"gcc -O3 -march=native -Wall -fPIC -shared {spmm_c} -o {so_path} -lm"
        
        print(f"Compilando kernel C para HBAG: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

setup(
    name='hbag',
    version='0.1.0',
    description='HBAG-Core SpMM Accelerated Kernel',
    packages=find_packages(),
    package_data={'hbag': ['libhbag.so']},
    cmdclass={'build_py': CustomBuildPy},
    install_requires=[
        'numpy',
        'scipy'
    ],
)
