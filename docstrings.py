"""
This script was used to remove all type-hints from the docstrings,
since the type hints exist in the code itself and Sphinx can pick them
up just fine. If something is enclosed in parenthesis inside a
docstring, immediately followed by a colon symbol, the code picks it
up.
"""

import glob
import os
import re
from pathlib import Path

import numpy as np


def list_directories(root_dir):
    res = []
    for path, directories, _ in os.walk(root_dir):
        for directory in directories:
            res.append(os.path.join(path, directory))
    return res


def list_python_files(directory):
    res = []
    for file in glob.glob(os.path.join(directory, '*.py')):
        res.append(file)
    return res


res = list_directories('src/osmg/')

not_backup = []

for thing in res:
    if '.~' not in thing:
        not_backup.append(thing)

not_backup.append('src/osmg')

# find all available filenames
files = {}
for thing in not_backup:
    files[thing] = list_python_files(thing)

pattern = r'\(*?\):'  # type: ignore (this is so silly)

for module in files:
    for path in files[module]:
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
                    print('~~~')
                    print(line)
                    print('~~~')
                    print()
