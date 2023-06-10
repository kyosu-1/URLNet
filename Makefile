format: ## Run pysen format
	poetry run pysen run format

lint: ## Run code static analyse
	poetry run pysen run lint

test: ## Run pytest
	poetry run pytest