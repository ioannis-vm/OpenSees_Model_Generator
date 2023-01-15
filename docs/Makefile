# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Convert all notebook .py files to jupyter notebooks using jupytext
source/notebooks/%.ipynb: source/notebooks/%.py
	@jupytext --to notebook $<

# html target
html: Makefile source/notebooks/*.ipynb
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# clear target, in case clean-build is required
clear:
	@rm -rf build  # remove the build directory
	@rm -rf source/_autosummary  # remove the autosummary directory
	@rm -rf source/notebooks/*.ipynb  # remove generated notebooks
