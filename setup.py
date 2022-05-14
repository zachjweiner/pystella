#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import setup, find_packages, Command

PACKAGE_NAME = "pystella"


def find_git_revision(tree_root):
    tree_root = Path(tree_root).resolve()

    if not tree_root.joinpath(".git").exists():
        return None

    from subprocess import run, PIPE, STDOUT
    result = run(["git", "rev-parse", "HEAD"], shell=False,
                 stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True,
                 cwd=tree_root)

    git_rev = result.stdout
    git_rev = git_rev.decode()
    git_rev = git_rev.rstrip()

    assert result.returncode is not None
    if result.returncode != 0:
        from warnings import warn
        warn("unable to find git revision")
        return None

    return git_rev


def write_git_revision(package_name):
    dn = Path(__file__).parent
    git_rev = find_git_revision(dn)
    text = 'GIT_REVISION = "%s"\n' % git_rev
    dn.joinpath(package_name, "_git_rev.py").write_text(text)


write_git_revision(PACKAGE_NAME)


class PylintCommand(Command):
    description = "run pylint on Python source files"
    user_options = [
        # The format is (long option, short option, description).
        ("pylint-rcfile=", None, "path to Pylint config file"),
    ]

    def initialize_options(self):
        setup_cfg = Path("setup.cfg")
        if setup_cfg.exists():
            self.pylint_rcfile = setup_cfg
        else:
            self.pylint_rcfile = None

    def finalize_options(self):
        if self.pylint_rcfile:
            assert Path(self.pylint_rcfile).exists()

    def run(self):
        command = ["pylint"]
        if self.pylint_rcfile is not None:
            command.append(f"--rcfile={self.pylint_rcfile}")
        command.append(PACKAGE_NAME)

        from glob import glob
        for directory in ["test", "examples", "."]:
            command.extend(glob(f"{directory}/*.py"))

        from subprocess import run
        run(command)


class Flake8Command(Command):
    description = "run flake8 on Python source files"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        command = ["flake8"]
        command.append(PACKAGE_NAME)

        from glob import glob
        for directory in ["test", "examples", "."]:
            command.extend(glob(f"{directory}/*.py"))

        from subprocess import run
        run(command)


setup(
    name=PACKAGE_NAME,
    version="2021.1",
    description="A code generator for grid-based PDE solving on CPUs and GPUs",
    long_description=open("README.rst", "rt").read(),
    install_requires=[
        "numpy",
        "pyopencl>=2020.2",
        "loopy>=2022.1",
    ],
    author="Zachary J Weiner",
    url="https://github.com/zachjweiner/pystella",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Code Generators",
    ],
    packages=find_packages(),
    python_requires=">=3",
    project_urls={
        "Documentation": "https://pystella.readthedocs.io/en/latest/",
        "Source": "https://github.com/zachjweiner/pystella",
    },
    cmdclass={
        "run_pylint": PylintCommand,
        "run_flake8": Flake8Command,
    },
)
