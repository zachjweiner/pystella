#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages, Command

PACKAGE_NAME = "pystella"


# authoritative version in pytools/__init__.py
def find_git_revision(tree_root):
    # Keep this routine self-contained so that it can be copy-pasted into
    # setup.py.

    from os.path import join, exists, abspath
    tree_root = abspath(tree_root)

    if not exists(join(tree_root, ".git")):
        return None

    from subprocess import Popen, PIPE, STDOUT
    p = Popen(["git", "rev-parse", "HEAD"], shell=False,
              stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True,
              cwd=tree_root)
    (git_rev, _) = p.communicate()

    git_rev = git_rev.decode()

    git_rev = git_rev.rstrip()

    retcode = p.returncode
    assert retcode is not None
    if retcode != 0:
        from warnings import warn
        warn("unable to find git revision")
        return None

    return git_rev


def write_git_revision(package_name):
    from os.path import dirname, join
    dn = dirname(__file__)
    git_rev = find_git_revision(dn)

    with open(join(dn, package_name, "_git_rev.py"), "w") as outf:
        outf.write('GIT_REVISION = "%s"\n' % git_rev)


write_git_revision(PACKAGE_NAME)


class PylintCommand(Command):
    description = "run pylint on Python source files"
    user_options = [
        # The format is (long option, short option, description).
        ("pylint-rcfile=", None, "path to Pylint config file"),
    ]

    def initialize_options(self):
        if os.path.exists("setup.cfg"):
            self.pylint_rcfile = "setup.cfg"
        else:
            self.pylint_rcfile = None

    def finalize_options(self):
        if self.pylint_rcfile:
            assert os.path.exists(self.pylint_rcfile)

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
    version="2020.1",
    description="A code generator for grid-based PDE solving on CPUs and GPUs",
    long_description=open("README.rst", "rt").read(),
    install_requires=[
        "numpy",
        "pyopencl>=2020.2",
        "loopy>=2020.2",
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
