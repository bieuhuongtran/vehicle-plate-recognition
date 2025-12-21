SHELL := /bin/bash

init:
	sudo apt install -y python3.12-venv
	python -m venv vpr-env

run:
	source ./vpr-env/bin/activate ; python .

test:
	curl -d "name=admin&shoesize=12" http://0.0.0.0:8080