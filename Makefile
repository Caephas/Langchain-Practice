# Variables
PYTHON = python
POETRY = poetry
ENV_FILE = .env

# Default rule
.DEFAULT_GOAL := help

# Help rule
help:
	@echo "Available commands:"
	@echo "  make setup         - Install dependencies using Poetry"
	@echo "  make run-agent     - Run the agents_basic.py script"
	@echo "  make run-chains    - Run chains_basics.py script"
	@echo "  make run-chat      - Run chat_model_basic.py script"
	@echo "  make run-prompt    - Run basic_prompt_template.py script"
	@echo "  make run-rag       - Run RAG scripts (basic example)"
	@echo "  make clean         - Clean Python cache files"
	@echo "  make env-check     - Check if .env file exists"

# Setup dependencies
setup:
	@echo "Setting up dependencies..."
	$(POETRY) install

# Run scripts
run-agent:
	@echo "Running agent script..."
	$(PYTHON) agents/agents_basic.py

run-chains:
	@echo "Running chains script..."
	$(PYTHON) chains/chains_basics.py

run-chat:
	@echo "Running chat model script..."
	$(PYTHON) chat/chat_model_basic.py

run-prompt:
	@echo "Running prompt template script..."
	$(PYTHON) prompt_templates/basic_prompt_template.py

run-rag:
	@echo "Running RAG basic example..."
	$(PYTHON) rag/rag_basics.py

# Clean cache
clean:
	@echo "Cleaning Python cache files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Check if .env exists
env-check:
	@echo "Checking for .env file..."
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Error: .env file is missing."; \
		exit 1; \
	fi
	@echo ".env file found."