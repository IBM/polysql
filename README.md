# PolySQL: Cross-Dialect SQL Evaluation Framework

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

PolySQL is a comprehensive evaluation framework for benchmarking text-to-SQL models across multiple SQL dialects. It enables fair comparison of models by evaluating them on native database backends (SQLite, DuckDB, PostgreSQL, MySQL, ClickHouse, BigQuery, Snowflake, etc.) and supports cross-dialect data conversion to measure the true "dialect gap" in text-to-SQL systems.

## Key Features

- **Multi-Dialect Support**: Evaluate on 8+ SQL dialects including SQLite, DuckDB, PostgreSQL, MySQL, ClickHouse, BigQuery, Snowflake, and Databricks
- **Cross-Dialect Evaluation**: Convert data between backends to measure dialect-specific model performance
- **Parallel Execution**: Run experiments across multiple models and dialects simultaneously
- **Comprehensive Metrics**: Execution accuracy, semantic equivalence, and cross-dialect robustness
- **Benchmark Support**: Compatible with BIRD, Spider, and other text-to-SQL benchmarks
- **Result Analysis**: Built-in visualization and analysis tools

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/polysql.git
cd polysql

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install .
```

### Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Add your API keys to `.env`:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
# Add database credentials as needed
```

### Download Datasets

```bash
# Download BIRD development set
curl -o dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip -d data/BIRD/
cd data/BIRD/dev_20240627 && unzip dev_databases.zip
```

### Run Your First Experiment

```bash
# Quick test with SQLite (no database setup required)
python src/polysql/evaluation/core/experiments.py \
  --models deepseek-v32:rits \
  --dialects sqlite \
  --datasets bird_dev \
  --n-examples 10 \
  --instructions-levels 12 \
  --workers 2
```

## Understanding Instruction Levels

Instruction levels control the prompting strategy (two-digit format `XY`):
- **X** = Chain-of-Thought (1=none, 2=with COT)
- **Y** = Dialect-specific instructions (1=generic, 2=dialect-specific, 3=with examples)

Common configurations:
- `11`: No COT, generic SQL instructions
- `22`: With COT, dialect-specific instructions (recommended)
- `23`: With COT, dialect-specific instructions + examples

## Results

Results are saved to `results/` directory with the following structure:

```
results/
├── 2026-01-19_15-30-00/
│   ├── gpt-4o_sqlite_bird_dev_22.json
│   ├── gpt-4o_duckdb_bird_dev_22.json
│   └── summary.csv
```

### Viewing Results

Use the Streamlit viewer:
```bash
streamlit run viewers/view_summary.py
```

## Cross-Dialect Evaluation

PolySQL supports evaluating queries generated for one dialect on data loaded into another dialect:

```bash
python src/polysql/evaluation/core/experiments.py \
  --models gpt-4o:openai \
  --dialects sqlite \
  --datasets bird_mini_dev_mysql \  # MySQL source data
  --n-examples 50 \
  --instructions-levels 22
```

This tests whether models can generate correct SQLite queries when the original benchmark data comes from MySQL.

## Architecture

```
src/polysql/
├── evaluation/
│   ├── backends/          # Database connectors and execution engines
│   │   ├── connectors/    # Native and cross-dialect connectors
│   │   ├── execution.py   # Query execution logic
│   │   └── connections.py # Database connection management
│   ├── config/            # Experiment configuration and dataset registry
│   ├── core/              # Core evaluation loop and model interface
│   ├── metrics/           # Result comparison and dialect metrics
│   ├── prompts/           # Dialect-specific prompt generation
│   ├── analysis/          # Result parsing and insights
│   └── utils/             # Shared utilities
└── utils/                 # General utilities
```

## Supported Databases

| Database | Native Eval | Cross-Dialect Source | Cross-Dialect Target |
|----------|-------------|----------------------|----------------------|
| SQLite | ✅ | ✅ | ✅ |
| DuckDB | ✅ | ✅ | ✅ |
| PostgreSQL | ✅ | ✅ | ✅ |
| MySQL | ✅ | ✅ | ✅ |
| ClickHouse | ✅ | ✅ | ✅ |
| BigQuery | ✅ | ✅ | ✅ |
| Snowflake | ✅ | ✅ | ✅ |
| Databricks | ✅ | ❌ | ✅ |

## Citation

If you use PolySQL in your research, please cite:

```bibtex
@article{polysql2026,
  title={PolySQL: Cross-Dialect SQL Evaluation for Text-to-SQL Benchmarks},
  author={Your Name and Co-Author Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026},
  url={https://arxiv.org/abs/XXXX.XXXXX}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built on top of:
- [Ibis](https://ibis-project.org/) for unified database connectivity
- [dlt](https://dlthub.com/) for cross-dialect data loading
- [SQLGlot](https://github.com/tobymao/sqlglot) for SQL parsing and transpilation
- [LiteLLM](https://github.com/BerriAI/litellm) for unified LLM API access

## Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com](mailto:your.email@example.com).
# polysql
