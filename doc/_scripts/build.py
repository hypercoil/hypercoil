# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Build documentation for each library using ``quartodoc``.
"""
import yaml
from io import StringIO
from pathlib import Path
from shutil import copy, rmtree
from subprocess import run

import quartodoc


def main():
    # Path to the root of the documentation
    root = Path(__file__).parent.parent
    # Clean rendered files
    rmtree(root / '_build', ignore_errors=True)
    # Read the file containing the library names
    with open(root / '_pkgidx.yml') as f:
        libraries = yaml.safe_load(f)
    # Read the YAML config template
    with open(root / '_quartodoc.yml') as f:
        template = f.read()
    # Iterate over each library
    for library, libmeta in libraries.items():
        rmtree(root / library / '.quarto', ignore_errors=True)
        # Create a YAML config for the library
        with open(root / library / '_quartodoc.yml') as f:
            libspec = f.read()
        libspec = libspec.format(
            __PKGNAME__=library,
            __PKGDESC__=libmeta['description'],
        )
        libspec = template.format(
            __PKGNAME__=library,
            __PKGDESC__=libmeta['description'],
            __PKGSPEC__=libspec,
        )
        _libspec = StringIO()
        _libspec.write(libspec)
        _libspec.seek(0)
        libspec = yaml.safe_load(_libspec)
        if libspec.get('static-resources'):
            static_src = root / '_static'
            static_dst = root / library / '_static'
            Path(static_dst).mkdir(exist_ok=True)
            for src in libspec['static-resources']:
                (static_dst / src).parent.mkdir(exist_ok=True, parents=True)
                dst = static_dst / src
                src = static_src / src
                copy(src, dst)
            del libspec['static-resources']
        if not (
            root / library / libspec['website']['navbar']['logo']
        ).exists():
            libspec['website']['navbar']['title'] = library
            del libspec['website']['navbar']['logo']
        # Build the documentation
        # Why doesn't builder.build work? We have to use the CLI instead,
        # but it's not clear why.
        # builder = quartodoc.Builder.from_quarto_config(libspec)
        # breakpoint()
        # builder.build()
        with open(root / library / '_quarto.yml', 'w+') as f:
            yaml.dump(libspec, f, sort_keys=False)
        run(['quartodoc', 'build'], cwd=root / library)
        run(
            ['quarto', 'render', '--cache-refresh', '--no-clean'],
            cwd=root / library,
        )
        (root / library / '_quarto.yml').unlink()
    run(['quarto', 'render', '--cache-refresh', '--no-clean'], cwd=root)


if __name__ == "__main__":
    main()
