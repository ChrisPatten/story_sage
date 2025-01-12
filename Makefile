# Declare all phony targets (targets that don't create files)
.PHONY: redis app setup

# Configuration variables
PYTHON_VERSION = 3.11.4    # Specific Python version required
VENV_NAME = story_sage_env # Virtual environment name

# setup_environment: Initializes Python environment using pyenv
# 1. Checks/installs pyenv
# 2. Installs specified Python version
# 3. Creates and configures virtual environment
# 4. Installs project dependencies
setup_environment:
	@echo "Checking for python environment"
	# Check and install pyenv if needed
	@if ! command -v pyenv &> /dev/null; then \
		echo "pyenv is not installed. Installing with brew install pyenv"; \
		brew install pyenv; \
	else \
		echo "pyenv is installed"; \
	fi
	# Check and install required Python version
	@echo "Checking for python version $(PYTHON_VERSION)"
	@if ! pyenv versions | grep -q $(PYTHON_VERSION); then \
		echo "Python version $(PYTHON_VERSION) is not installed. Installing with pyenv install $(PYTHON_VERSION)"; \
		pyenv install -s $(PYTHON_VERSION); \
	else \
		echo "Python version $(PYTHON_VERSION) is installed"; \
	fi
	# Create virtual environment if it doesn't exist
	@echo "Checking for virtual environment $(VENV_NAME)"
	@if ! pyenv virtualenvs | grep -q $(VENV_NAME); then \
		echo "Virtual environment $(VENV_NAME) is not installed. Creating with pyenv virtualenv -f $(PYTHON_VERSION) $(VENV_NAME)"; \
		pyenv virtualenv -f $(PYTHON_VERSION) $(VENV_NAME); \
	else \
		echo "Virtual environment $(VENV_NAME) is installed"; \
	fi
	# Install project dependencies
	@eval "$$(pyenv init -)" && \
	eval "$$(pyenv virtualenv-init -)" && \
	pip install -r requirements.txt

# install_redis: Checks and installs Redis using Homebrew if not present
install_redis:
	@echo "Checking if redis is installed"
	@if ! command -v redis-server &> /dev/null; then \
		echo "Redis is not installed. Installing with brew install redis"; \
		brew install redis; \
	else \
		echo "Redis is installed"; \
	fi

# redis: Manages Redis server instance
# 1. Creates config file if missing
# 2. Starts Redis in a screen session if not running
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

# app: Runs the main application
# 1. Ensures Redis is running
# 2. Activates virtual environment
# 3. Starts the application
app: redis
	@echo "Activating pyenv virtual environment and running script"
	@eval "$$(pyenv init -)" && \
	eval "$$(pyenv virtualenv-init -)" && \
	pyenv activate story_sage_env && \
	python3 app.py

# setup: Main setup target that initializes the entire application
# 1. Sets up Python environment
# 2. Installs Redis
# 3. Creates initial configuration files
setup:
	@echo "Installing dependencies and ensuring environment is configured correctly"
	@make setup_environment
	@make install_redis
	@if [ ! -f config.yml ]; then \
		echo "Creating config.yml"; \
		cp config.example.yml config.yml; \
	fi
	@echo "Setup complete. Update config.yml as necessary and ensure a series.yml file is present in the root directory"

test_model:
	@echo "Running test queries"
	@eval "$$(pyenv init -)" && \
	eval "$$(pyenv virtualenv-init -)" && \
	pyenv activate story_sage_env && \
	python3 tests/quality_test.py

test:
	@echo "Running all tests"
	@eval "$$(pyenv init -)" && \
	eval "$$(pyenv virtualenv-init -)" && \
	pyenv activate story_sage_env && \
	pytest story_sage/tests/

vulture:
	@echo "Running all tests"
	@eval "$$(pyenv init -)" && \
	eval "$$(pyenv virtualenv-init -)" && \
	pyenv activate story_sage_env && \
	vulture story_sage/ --exclude story_sage/tests/