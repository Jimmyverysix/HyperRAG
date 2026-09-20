PYTHON ?= python
REPO_ROOT := $(abspath ../..)
RDIR := research/path_consistent_negative_learning
MODULE := research.path_consistent_negative_learning
CONFIG := $(RDIR)/configs/www_revision/proxy_main.json
ARTIFACT_ROOT := $(RDIR)/artifacts/www_revision
HISTORICAL_ROOT := $(RDIR)/artifacts
PAPER_ROOT := $(RDIR)/paper/www2027

DATA_SOURCE ?= /root/hyperrag_pcneg/data_sources/WikiTopics_QE
AUDIT_RAW ?= /root/hyperrag_pcneg/runs/wikitopics_audit
WORK_ROOT ?= /root/hyperrag_pcneg/www_revision
PROXY_DATA := $(WORK_ROOT)/data
RUN_ROOT := $(WORK_ROOT)/runs
MANIFEST_ROOT := $(WORK_ROOT)/manifests
LAMBDA_ROOT := $(WORK_ROOT)/lambda_selection
GPU_IDS ?= 0 1 2 3 4 5

.PHONY: test verify-history audit prepare sweep-manifest sweep select-lambda \
	test-manifests main-experiments aggregate figures paper-results paper all

test:
	cd $(REPO_ROOT) && $(PYTHON) -m unittest discover -s $(RDIR)/tests -v

verify-history:
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.verify_historical_artifacts \
		--artifacts-dir $(HISTORICAL_ROOT) \
		--output $(ARTIFACT_ROOT)/provenance/historical_verification.json

audit:
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.aggregate_audit_distribution \
		--input-root $(AUDIT_RAW) \
		--historical-summary $(HISTORICAL_ROOT)/audit/combined_summary.json \
		--output-dir $(ARTIFACT_ROOT)/audit

prepare:
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.prepare_www_proxy_data \
		--config $(CONFIG) --source-root $(DATA_SOURCE) \
		--output-root $(PROXY_DATA) --workers 6

sweep-manifest:
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.build_www_job_manifest \
		--phase selection --config $(CONFIG) --data-root $(PROXY_DATA) \
		--run-root $(RUN_ROOT) --output $(MANIFEST_ROOT)/selection.json

sweep:
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.run_experiment_queue \
		--manifest $(MANIFEST_ROOT)/selection.json --gpus $(GPU_IDS)

select-lambda:
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.select_all_lambdas \
		--config $(CONFIG) --run-root $(RUN_ROOT) --output-root $(LAMBDA_ROOT)

test-manifests:
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.build_www_job_manifest \
		--phase test-checkpoints --config $(CONFIG) --data-root $(PROXY_DATA) \
		--run-root $(RUN_ROOT) --lambda-root $(LAMBDA_ROOT) \
		--output $(MANIFEST_ROOT)/test_checkpoints.json
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.build_www_job_manifest \
		--phase test-controls --config $(CONFIG) --data-root $(PROXY_DATA) \
		--run-root $(RUN_ROOT) --lambda-root $(LAMBDA_ROOT) \
		--output $(MANIFEST_ROOT)/test_controls.json

main-experiments:
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.run_experiment_queue \
		--manifest $(MANIFEST_ROOT)/test_checkpoints.json --gpus $(GPU_IDS)
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.run_experiment_queue \
		--manifest $(MANIFEST_ROOT)/test_controls.json --gpus $(GPU_IDS)

aggregate:
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.aggregate_www_proxy_results \
		--config $(CONFIG) --run-root $(RUN_ROOT) --lambda-root $(LAMBDA_ROOT) \
		--audit $(ARTIFACT_ROOT)/audit/path_conflict_audit.json \
		--output-dir $(ARTIFACT_ROOT)/proxy

figures:
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).figures.gen_www_method \
		--output-stem $(ARTIFACT_ROOT)/figures/method_overview
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).figures.gen_www_conflict \
		--audit $(ARTIFACT_ROOT)/audit/path_conflict_audit.json \
		--output-stem $(ARTIFACT_ROOT)/figures/path_conflict
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).figures.gen_www_lambda \
		--selection-csv $(LAMBDA_ROOT)/lambda_selection.csv \
		--audit $(ARTIFACT_ROOT)/audit/path_conflict_audit.json \
		--output-stem $(ARTIFACT_ROOT)/figures/lambda_sensitivity
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).figures.gen_www_tradeoff \
		--summary $(ARTIFACT_ROOT)/proxy/proxy_main_summary.json \
		--output-stem $(ARTIFACT_ROOT)/figures/coverage_ranking_tradeoff

paper-results:
	cd $(REPO_ROOT) && $(PYTHON) -m $(MODULE).scripts.generate_paper_results \
		--historical $(ARTIFACT_ROOT)/provenance/historical_verification.json \
		--audit $(ARTIFACT_ROOT)/audit/path_conflict_audit.json \
		--proxy-summary $(ARTIFACT_ROOT)/proxy/proxy_main_summary.json \
		--proxy-table $(ARTIFACT_ROOT)/proxy/proxy_main_table.csv \
		--selection $(LAMBDA_ROOT)/lambda_selection.csv \
		--output-dir $(PAPER_ROOT)/generated

paper:
	cd $(REPO_ROOT)/$(PAPER_ROOT) && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex

all: test verify-history audit prepare sweep-manifest sweep select-lambda \
	test-manifests main-experiments aggregate figures paper-results paper
