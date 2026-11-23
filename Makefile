PYTHON ?= python

.PHONY: classifier-up classifier-down classify-pool eval-ch10 export-onnx train-ch10 data-prep-ch10 replay-threshold

classifier-up:
	$(PYTHON) scripts/classifier_service.py start

classifier-down:
	$(PYTHON) scripts/classifier_service.py stop

classify-pool:
	$(PYTHON) scripts/classify_pool.py --min-batch 10

eval-ch10:
	$(PYTHON) scripts/evaluate_classifier.py

export-onnx:
	$(PYTHON) scripts/export_onnx.py

train-ch10:
	$(PYTHON) scripts/train_classifier.py

data-prep-ch10:
	$(PYTHON) scripts/data_prep_ch10.py

replay-threshold:
	$(PYTHON) scripts/scan_threshold_replay.py
