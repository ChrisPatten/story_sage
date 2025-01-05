.PHONY: redis app setup

# Define your Python version and virtual environment name
PYTHON_VERSION = 3.11.4
VENV_NAME = story_sage_env

# Target to initialize pyenv and create a virtual environment
setup_environment:
	@echo "Checking for python environment"
	@if ! command -v pyenv &> /dev/null; then \
		echo "pyenv is not installed. Installing with brew install pyenv"; \
		brew install pyenv; \
	else \
		echo "pyenv is installed"; \
	fi
	@echo "Checking for python version $(PYTHON_VERSION)"
	@if ! pyenv versions | grep -q $(PYTHON_VERSION); then \
		echo "Python version $(PYTHON_VERSION) is not installed. Installing with pyenv install $(PYTHON_VERSION)"; \
		pyenv install -s $(PYTHON_VERSION); \
	else \
		echo "Python version $(PYTHON_VERSION) is installed"; \
	fi
	@echo "Checking for virtual environment $(VENV_NAME)"
	@if ! pyenv virtualenvs | grep -q $(VENV_NAME); then \
		echo "Virtual environment $(VENV_NAME) is not installed. Creating with pyenv virtualenv -f $(PYTHON_VERSION) $(VENV_NAME)"; \
		pyenv virtualenv -f $(PYTHON_VERSION) $(VENV_NAME); \
	else \
		echo "Virtual environment $(VENV_NAME) is installed"; \
	fi
	@eval "$$(pyenv init -)" && \
	eval "$$(pyenv virtualenv-init -)" && \
	pip install -r requirements.txt

install_redis:
	@echo "Checking if redis is installed"
	@if ! command -v redis-server &> /dev/null; then \
		echo "Redis is not installed. Installing with brew install redis"; \
		brew install redis; \
	else \
		echo "Redis is installed"; \
	fi


redis:
	@if [ ! -f redis_config.conf ]; then \
		echo "Creating redis_config.conf"; \
		cp redis_config.example.conf redis_config.conf; \
	fi
	@if ! screen -list | grep -q "redis"; then \
		screen -dmS redis redis-server redis_config.conf; \
		echo "Started new redis screen session with config file"; \
	else \
		echo "Redis screen session already running"; \
	fi

app: redis
	@echo "Activating pyenv virtual environment and running script"
	@eval "$$(pyenv init -)" && \
	eval "$$(pyenv virtualenv-init -)" && \
	pyenv activate story_sage_env && \
	python3 app.py

setup:
	@echo "Installing dependencies and ensuring environment is configured correctly"
	@make setup_environment
	@make install_redis
	@if [ ! -f config.yml ]; then \
		echo "Creating config.yml"; \
		cp config.example.yml config.yml; \
	fi
	@echo "Setup complete. Update config.yml as necessary and ensure a series.yml file is present in the root directory"

test:
	@echo "Running tests"
	@eval "$$(pyenv init -)" && \
	eval "$$(pyenv virtualenv-init -)" && \
	pyenv activate story_sage_env && \
	python3 tests/quality_test.py

