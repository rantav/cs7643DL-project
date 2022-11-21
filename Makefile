setup:
	@echo "Setting up environment..."
	rm -rf .venv
	python3 -m venv .venv
	source .venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt
	mkdir -p saved-models
	@echo "Done."

update-requirements:
	@echo "Updating requirements..."
	source .venv/bin/activate && \
		pip install -r requirements.txt
	@echo "Done."

data:
	@echo "Downloading data..."
		wget -P /tmp https://storage.googleapis.com/dl-project-data/data-by-artist.zip
	unzip /tmp/data-by-artist.zip -d .
	@echo "Done."

jupyter:
	@echo "Starting Jupyter Notebook..."
	source .venv/bin/activate && \
		jupyter-lab

tensorboard:
	@echo "Starting Tensorboard..."
	source .venv/bin/activate && \
		tensorboard --logdir=runs

run-classifier-phase1:
	@echo "Running classifier phase 1..."
	source .venv/bin/activate && \
		python3 style_classifier.py

run-style-transfer:
	@echo "Running style transfer..."
	source .venv/bin/activate && \
		python3 style_transfer.py