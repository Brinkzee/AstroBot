.PHONY: classifier-up classifier-down classify-pool eval-ch10 export-onnx train-ch10 data-prep-ch10 replay-threshold

classifier-up:
	python scripts/classifier_service.py start

classifier-down:
	python scripts/classifier_service.py stop

classify-pool:
	python scripts/classify_pool.py --min-batch 10

eval-ch10:
	python scripts/evaluate_classifier.py

export-onnx:
	python scripts/export_onnx.py

train-ch10:
	python scripts/train_classifier.py

data-prep-ch10:
	python scripts/data_prep_ch10.py

replay-threshold:
	python scripts/scan_threshold_replay.py
