SHELL := /bin/bash

init:
	git submodule update --init --recursive
	sudo apt install -y python3.12-venv
	python -m venv vpr-env
	source ./vpr-env/bin/activate ; \
	pip install -r ./VehiclePlateRecognition/yolov5/requirements.txt ; \
	pip install -r requirements.txt

run:
	source ./vpr-env/bin/activate ; python .

test:
	curl -d "name=admin&shoesize=12" http://0.0.0.0:8080