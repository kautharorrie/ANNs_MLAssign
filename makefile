VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

run: $(VENV)/bin/activate
	$(PYTHON) app.py


$(VENV)/bin/activate: requirements.txt
	vitualenv env $(VENV)
	$(PIP) install -r requirements.txt
clean: 
	rm -rf __pycache__
	rm -rf $(VENV)
	

