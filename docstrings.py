"""
Remove type hints from docstrings.

This script was used to remove all type-hints from the docstrings,
since the type hints exist in the code itself and Sphinx can pick them
up just fine. If something is enclosed in parenthesis inside a
docstring, immediately followed by a colon symbol, the code picks it
up.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np


def list_directories(root_dir: str) -> list[Path]:
    """
    List directories.

    Returns:
      List of directories.
    """
    res = []
    for path, directories, _ in os.walk(root_dir):
        for directory in directories:
            res.append(Path(path) / directory)  # noqa: PERF401
    return res


def list_python_files(directory: str) -> list[Path]:
    """
    List all python files.

    List all Python files in the specified directory and
    subdirectories.

    Args:
        directory (str): The directory to search in.

    Returns:
    -------
        list[str]: A list of paths to Python files.
    """
    return [str(file) for file in Path(directory).rglob('*.py')]


res = list_directories('src/osmg/')

not_backup = []

for thing in res:
    if '.~' not in thing:
        not_backup.append(thing)  # noqa: PERF401

not_backup.append('src/osmg')

# find all available filenames
files = {}
for thing in not_backup:
    files[thing] = list_python_files(thing)

pattern = r'\(*?\):'  # type: ignore (this is so silly)

for paths in files.values():
    for path in paths:
        contents = Path(path).read_text(encoding='utf-8')
        if contents.startswith('"""'):
            contents = '\n\n' + contents
        contents_spl = np.array(contents.split('"""'))
        contents_docstr = contents_spl[1::2]
        for thing in contents_docstr:
            lines = thing.split('\n')
            for line in lines:
                if '>>>' in line:
                    continue
                if '...' in line:
                    continue
                if '(most recent call last):' in line:
                    continue
                match = re.search(pattern, line)
                if match:
                    print('~~~')  # noqa: T201
                    print(line)  # noqa: T201
                    print('~~~')  # noqa: T201
                    print()  # noqa: T201
