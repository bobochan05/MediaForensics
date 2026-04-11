PYTHON ?= python

.PHONY: install backend frontend test

install:
	$(PYTHON) -m pip install -r requirements.txt

backend:
	$(PYTHON) app.py

frontend:
	cd frontend/dashboard-app && npm install && npm run dev

test:
	$(PYTHON) -m pytest -q

