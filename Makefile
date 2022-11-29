setup:
	@echo "Setting up environment..."
	rm -rf .venv
	python3 -m venv .venv
	. .venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt
	mkdir -p saved-models
	@echo "Done."

update-requirements:
	@echo "Updating requirements..."
	. .venv/bin/activate && \
		pip install -r requirements.txt
	@echo "Done."

data:
	@echo "Downloading data..."
		wget -P ./ https://storage.googleapis.com/dl-project-data/by-artist-4artists-256.zip
	unzip ./by-artist-4artists-256.zip -d .
	@echo "Done."

upload-data:
	@echo "Uploading data..."
	rm -rf by-artist-4artists-256.zip
	zip -r by-artist-4artists-256.zip data/by-artist-4artists-256/
	zip -ur by-artist-4artists-256.zip data/content

	@echo "Done. Now you need to upload the file by-artist-4artists-256.zip to Google"


jupyter:
	@echo "Starting Jupyter Notebook..."
	. .venv/bin/activate && \
		jupyter-lab

tensorboard:
	@echo "Starting Tensorboard..."
	. .venv/bin/activate && \
		tensorboard --logdir=runs

run-train-style-classifier:
	@echo "Running classifier phase training..."
	source .venv/bin/activate && \
		python3 style_classifier.py  \
			 --task train

run-classify-style:
	@echo "Classifying styles..."
	source .venv/bin/activate && \
		python3 style_classifier.py  \
			--task classify

run-classify-style-after-style-transfer:
	@echo "Classifying styles..."
	source .venv/bin/activate && \
		python3 style_classifier.py  \
			--task classify \
			--test_directory data/output/style_transfered256_style_weight_1000000


run-style-transfer:
	@echo "Running style transfer..."
	source .venv/bin/activate && \
		python3 style_transfer.py \
			--image_size=256 \
			--output_dir=data/output/style_transfered256
