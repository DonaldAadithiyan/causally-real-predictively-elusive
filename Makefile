# Common entry points. All targets run from the repository root.
PY ?= python

.PHONY: install data smoke test tables figures reproduce-all \
        reproduce-train reproduce-arrays reproduce-phase34 reproduce-e3 reproduce-e4 reproduce-e5 \
        reproduce-e6 reproduce-scorer reproduce-f reproduce-g reproduce-reports

install:            ## pinned environment + editable package
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

data:               ## download ETTh1/ETTh2 and hash-check every dataset present
	$(PY) scripts/download_data.py

smoke: data         ## tiny end-to-end run on CPU (~30 s; target < 10 min)
	$(PY) scripts/smoke_test.py --out smoke_out

test:               ## leakage, paper numbers, config constants, smoke
	$(PY) -m pytest -q

tables:             ## Tables 2-4 from results/ -> paper/tables/
	$(PY) -m paper.make_tables

figures:            ## Figure 2 from results/ -> paper/figures/fig_dissociation_col.pdf
	$(PY) -m paper.make_fig2

reproduce-all:      ## every camera-ready stage in order (see scripts/reproduce_all.sh for times)
	bash scripts/reproduce_all.sh all

reproduce-%:
	bash scripts/reproduce_all.sh $*
