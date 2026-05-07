# Contributing to PolySQL

Thank you for your interest in contributing to PolySQL! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/polysql.git`
3. Create a virtual environment: `python -m venv .venv`
4. Install dependencies: `pip install -e ".[dev]"`
5. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Guidelines

### Code Style

- Use [Ruff](https://github.com/astral-sh/ruff) for formatting and linting
- Format code before committing: `ruff format .`
- Maximum line length: 100 characters
- Use type hints for all function signatures

### Testing

- Write tests for all new features
- Run tests before submitting PR: `pytest`
- Ensure all tests pass: `pytest -v`
- Add integration tests for new database connectors

### Documentation

- Update README.md if adding new features
- Write clear docstrings for all public functions
- Keep docstrings concise (one line is often sufficient)
- Update CLAUDE.md if changing development workflows

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove)
- Keep first line under 72 characters
- Example: `Add ClickHouse connector for cross-dialect evaluation`

## Adding a New Database Dialect

To add support for a new SQL dialect:

1. **Add Native Connector** (`src/polysql/evaluation/backends/connectors/native/<dialect>.py`)
2. **Add Cross-Dialect Connectors** (`cross_dialect/sqlite_to_<dialect>.py`, etc.)
3. **Update Factory** (`connectors/factory.py`)
4. **Add DDL Extraction** (`backends/connections.py`)
5. **Add Prompt Templates** (`prompts/sql.py`)
6. **Update Supported Dialects** (`core/model.py`)
7. **Add Tests** (`tests/evaluation/unit/test_<dialect>_connector.py`)

See CLAUDE.md for detailed instructions.

## Submitting Changes

1. Ensure all tests pass
2. Format code with Ruff
3. Commit your changes with clear messages
4. Push to your fork
5. Open a Pull Request with:
   - Clear description of changes
   - Motivation for the change
   - Any relevant issue numbers

## Code of Conduct

- Be respectful and constructive
- Focus on technical merit
- Welcome newcomers and beginners
- Provide helpful feedback

## Questions?

Open an issue on GitHub or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
