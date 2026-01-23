# =============================================================================
# CodeSentinel Makefile
# Common commands for development and deployment
# =============================================================================

.PHONY: help install dev run test lint format clean docker-build docker-run docker-stop logs

# Default target
help:
	@echo "CodeSentinel - Available Commands"
	@echo "================================="
	@echo ""
	@echo "Development:"
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Run in development mode"
	@echo "  make run         - Run the application"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linter"
	@echo "  make format      - Format code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-run     - Run with Docker Compose"
	@echo "  make docker-stop    - Stop Docker containers"
	@echo "  make docker-logs    - View container logs"
	@echo "  make docker-shell   - Shell into the container"
	@echo ""
	@echo "Production:"
	@echo "  make deploy      - Deploy to production"
	@echo "  make logs        - View application logs"
	@echo "  make clean       - Clean up temporary files"
	@echo ""

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

dev:
	@echo "🚀 Starting development server..."
	KMP_DUPLICATE_LIB_OK=TRUE ENVIRONMENT=development LOG_LEVEL=DEBUG streamlit run app.py

run:
	@echo "🚀 Starting application..."
	KMP_DUPLICATE_LIB_OK=TRUE streamlit run app.py --server.port=8501 --server.address=0.0.0.0

test:
	@echo "🧪 Running all tests..."
	python -m pytest tests/ -v --tb=short

test-unit:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/unit/ -v --tb=short -m "not integration"

test-integration:
	@echo "🧪 Running integration tests..."
	python -m pytest tests/integration/ -v --tb=short -m "integration"

test-cov:
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated in htmlcov/"

test-fast:
	@echo "🧪 Running fast tests only..."
	python -m pytest tests/unit/ -v --tb=short -x -q

lint:
	@echo "🔍 Running linter..."
	python -m pylint *.py --disable=C0114,C0115,C0116 --max-line-length=120

format:
	@echo "✨ Formatting code..."
	python -m black *.py --line-length=120
	python -m isort *.py

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build
	docker build -f Dockerfile.sandbox -t codesentinel-sandbox:latest .
	@echo "✅ Docker images built"

docker-run:
	@echo "🐳 Starting with Docker Compose..."
	docker-compose up -d
	@echo "✅ Application running at http://localhost:8501"

docker-stop:
	@echo "🛑 Stopping containers..."
	docker-compose down
	@echo "✅ Containers stopped"

docker-logs:
	@echo "📋 Container logs:"
	docker-compose logs -f

docker-shell:
	@echo "🐚 Opening shell in container..."
	docker-compose exec app /bin/bash

# -----------------------------------------------------------------------------
# Production
# -----------------------------------------------------------------------------

deploy: docker-build
	@echo "🚀 Deploying to production..."
	ENVIRONMENT=production docker-compose up -d
	@echo "✅ Deployed to production"

logs:
	@echo "📋 Application logs:"
	@if [ -f ./logs/codesentinel.log ]; then \
		tail -f ./logs/codesentinel.log; \
	else \
		echo "No log file found. Using Docker logs..."; \
		docker-compose logs -f app; \
	fi

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	@echo "✅ Cleanup complete"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

check-env:
	@echo "🔍 Checking environment..."
	@if [ -f .env ]; then \
		echo "✅ .env file exists"; \
	else \
		echo "❌ .env file missing! Copy from env.example"; \
		exit 1; \
	fi
	@python -c "from config import settings; errors = settings.validate(); print('\\n'.join(errors) if errors else '✅ Configuration valid')"

health:
	@echo "🏥 Health check..."
	@curl -s http://localhost:8501/_stcore/health || echo "❌ Application not running"
