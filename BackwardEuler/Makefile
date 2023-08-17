format:
	isort --profile black -l 100 *.py
	autopep8 --in-place --aggressive --aggressive *.py      
	black -l 100 *.py

run:
	python3 main.py


condor:
	condor_submit cluster/condor.submit


uml:
	pyreverse -o png FOM.py
	mv classes.png FOM.png
	pyreverse -o png ROM.py
	mv classes.png ROM.png