SHELL := /bin/bash

run:
	python .

test:
	curl -d "name=admin&shoesize=12" http://0.0.0.0:8080