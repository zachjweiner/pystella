#!/usr/bin/env python

from pathlib import Path
from setuptools import setup


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
    (dn / package_name / "_git_rev.py").write_text(text)


write_git_revision("pystella")

setup()
