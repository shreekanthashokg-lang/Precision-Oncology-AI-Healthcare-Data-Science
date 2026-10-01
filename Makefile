.PHONY: install clinical clinvar-real data-genomics data-vision preprocess train-genomics train-vision app clean

install:
	pip install -r requirements.txt

clinical:
	python -m src.clinical.wisconsin

clinvar-real:
	python -m src.genomics.clinvar_real

data-genomics:
	python data/scripts/download_genomics.py --mode real_curated

data-vision:
	python data/scripts/download_histopathology.py --dataset pcam

preprocess: data-genomics
	python data/scripts/preprocess.py --task genomics

train-genomics:
	python -m src.genomics.train

train-vision:
	python -m src.vision.train

train: train-genomics train-vision

app:
	streamlit run app/streamlit_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf results/logs/*.log
