"""
Test that the code that generates the tutorial notebooks runs without
producing any errors.

"""

import os


def test_2_define_a_model() -> None:
    from docs.source.notebooks import doc_2_define_a_model  # noqa: PLC0415


def test_3_run_an_analysis() -> None:
    os.chdir('docs/source/notebooks')
    from docs.source.notebooks import doc_3_run_an_analysis  # noqa: PLC0415
