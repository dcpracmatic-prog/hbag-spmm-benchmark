from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import subprocess
import os

class BuildWithGCC(build_py):
    def run(self):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        src_c = os.path.join(base_dir, 'src', 'spmm.c')
        target_dir = os.path.join(base_dir, 'hbag')
        os.makedirs(target_dir, exist_ok=True)
        so_path = os.path.join(target_dir, 'libhbag.so')

        print(f"[*] Compilando kernel C para HBAG...")
        cmd = f"gcc -O3 -march=native -Wall -fPIC -shared '{src_c}' -o '{so_path}' -lm"
        res = subprocess.run(cmd, shell=True, capture_output=True)
        
        if res.returncode != 0:
            cmd_generic = f"gcc -O3 -Wall -fPIC -shared '{src_c}' -o '{so_path}' -lm"
            subprocess.run(cmd_generic, shell=True, check=True)

        super().run()

setup(
    name='hbag',
    version='0.1.0',
    packages=find_packages(),
    package_data={'hbag': ['libhbag.so']},
    include_package_data=True,
    cmdclass={'build_py': BuildWithGCC},
    install_requires=['numpy', 'scipy']
)
