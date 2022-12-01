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

data-download:
	@echo "Downloading data..."
	rm -f dl-project-data.zip
	wget -P ./ https://storage.googleapis.com/dl-project-data/dl-project-data.zip
	unzip ./dl-project-data.zip -d .
	@echo "Done."

data-upload:
	@echo "Zipping data..."
	rm -f dl-project-data.zip
	zip -qr dl-project-data.zip data/by-artist-4artists-256/
	zip -qur dl-project-data.zip data/by-content
	@echo "Done. Now you need to upload the file dl-project-data.zip to Google"


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
		python style_classifier.py  \
			 --task train

run-classify-style:
	@echo "Classifying styles..."
	source .venv/bin/activate && \
		python style_classifier.py  \
			--task classify

run-classify-style-after-style-transfer:
	@echo "Classifying styles..."
	source .venv/bin/activate && \
		python style_classifier.py  \
			--task classify \
			--test_directory data/output/style_transfered256_style_weight_1000000


run-style-transfer:
	@echo "Running style transfer..."
	source .venv/bin/activate && \
		python style_transfer.py \
			--image_size=256 \
			--output_dir=data/output/style_transfered

run-experiments:
	@echo "Running experiments..."
	source .venv/bin/activate && \
		python experiment.py \
			--image_size=256 \
			--output_dir=data/output/style_transfered \
			--images_per_artist=5 \
			--images_per_class=5 \
			--num_steps=300 \
			--style_weight 10000 50000 100000 500000 1000000

gcp-ssh:
	gcloud compute ssh \
		--zone "us-east4-c" "deeplearning-1-vm"  \
		--project "dl-project-368810"

gcp-copy-results:
	gcloud compute scp \
		--project dl-project-368810 \
		--zone us-east4-c \
		--recurse deeplearning-1-vm:~/cs7643DL-project/data/output/style_transfered data/output/
