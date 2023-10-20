format:
	isort --profile black -l 100 *.py
	autopep8 --in-place --aggressive --aggressive *.py      
	black -l 100 *.py

run:
	python3 main_iROM.py

iROM: 
	python3 main_iROM.py

ROM:
	python3 main_ROM.py

condor:
	condor_submit cluster/condor.submit

uml:
	pyreverse -o png FOM.py
	mv classes.png FOM.png
	pyreverse -o png ROM.py
	mv classes.png ROM.png
	pyreverse -o png iROM.py
	mv classes.png iROM.png

eval:
	python3 evaluate_iROM.py

plot:
	python3 plot_after_eval.py