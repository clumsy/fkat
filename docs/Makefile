# Minimal makefile for Sphinx documentation
# make html
# OR
# make livehtml

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    = -W
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = fkat
SOURCEDIR     = .
BUILDDIR      = _build/html
VERSION       := $(shell python -c "import fkat; print(fkat.__version__)")

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@echo "Deleting build directory"
	rm -rf "$(BUILDDIR)"

.PHONY: help Makefile clean livehtml


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# optional live version
livehtml:
	sphinx-autobuild --watch ../fkat --watch . "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
