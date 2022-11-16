setup:
	@echo "Setting up environment..."
	rm -rf .venv
	python3 -m venv .venv
	source .venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt
	@echo "Done."

update-requirements:
	@echo "Updating requirements..."
	source .venv/bin/activate && \
		pip install -r requirements.txt
	@echo "Done."
