SHELL := /bin/bash

init:
	git submodule update --init --recursive
	sudo apt install -y python3.12-venv
	python -m venv vpr-env
	source ./vpr-env/bin/activate ; \
	pip install -r ./VehiclePlateRecognition/model/yolov5/requirements.txt ; \
	pip install -r requirements.txt

run:
	source ./vpr-env/bin/activate ; python .

test-post:
	curl -F "file=@./test/image.png" http://0.0.0.0:8080

test-1:
	source ./vpr-env/bin/activate ; python ./test/test1.py

test-oop:
	source ./vpr-env/bin/activate ; python ./test/test-oop.py