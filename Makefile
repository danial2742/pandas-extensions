VIRTUALENV_DIR=$(shell pwd)/.pext-virtualenv
EXEC_IN_VENV=. $(VIRTUALENV_DIR)/bin/activate &&

virtualenv:
	@echo "Creating virtual env in $(VIRTUALENV_DIR)..."
	@(test -d $(VIRTUALENV_DIR) && echo "Virtual env directory exists.") || virtualenv $(VIRTUALENV_DIR)
