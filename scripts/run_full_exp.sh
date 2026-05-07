.venv/bin/python /Users/ype/Code/NL2DSL/src/nl2dsl/evaluation/core/experiments.py \
--model deepseek-v32:rits gpt-oss-20b:rits gpt-oss-120b:rits claude-haiku-4-5:open-ai claude-sonnet-4-5:open-ai deepseek-coder-33b-instruct:rits llama-4-maverick:rits llama-4-scout-17b-16e-instruct:rits llama-3-1-8b-instruct:rits llama-3-3-70b-instruct:rits devstral-small-2507:rits qwen3-8b:rits granite-3-3-8b-instruct:rits deepseek-v2-5:rits llama-3-1-405b-instruct-fp8:rits qwen2-5-72b-instruct:rits \
--n-examples 100 \
--dialects sqlite postgres mysql clickhouse bigquery snowflake \
--instructions-levels 12 \
--workers 1 \
--skip-data-load \
--datasets bird_mini_dev_sqlite archer_dev_s_and_c spider_dev

.venv/bin/python /Users/ype/Code/NL2DSL/src/nl2dsl/evaluation/core/experiments.py \
--model deepseek-v32:rits gpt-oss-20b:rits gpt-oss-120b:rits claude-haiku-4-5:open-ai claude-sonnet-4-5:open-ai deepseek-coder-33b-instruct:rits llama-4-maverick:rits llama-4-scout-17b-16e-instruct:rits llama-3-1-8b-instruct:rits llama-3-3-70b-instruct:rits devstral-small-2507:rits qwen3-8b:rits granite-3-3-8b-instruct:rits deepseek-v2-5:rits llama-3-1-405b-instruct-fp8:rits qwen2-5-72b-instruct:rits \
--dialects sqlite postgres mysql \
--instructions-levels 12 \
--workers 2 \
--skip-data-load \
--datasets bird_mini_dev_sqlite bird_mini_dev_mysql bird_mini_dev_postgres \
--n-examples 1000 

# debugging
.venv/bin/python /Users/ype/Code/NL2DSL/src/nl2dsl/evaluation/core/experiments.py \
--model deepseek-v32:rits \
--dialects mysql \
--instructions-levels 12 \
--workers 2 \
--skip-data-load \
--datasets bird_mini_dev_sqlite \
--n-examples 1000 


.venv/bin/python /Users/ype/Code/NL2DSL/src/nl2dsl/evaluation/analysis/enrich_results.py --enable-transpilation --results-dir results/2025-12-14_13-16-47_hopeful_kare --max-rows 1000
# .venv/bin/python /Users/ype/Code/NL2DSL/src/nl2dsl/evaluation/core/experiments.py --no-cache --model deepseek-v2-5:rits llama-3-1-405b-instruct-fp8:rits deepseek-v32:rits qwen2-5-72b-instruct:rits --n-examples 100 --dialects sqlite --instructions-levels 12 --workers 2 --datasets bird_mini_dev_sqlite archer_dev_s_and_c spider_dev bird_mini_dev_mysql bird_mini_dev_postgres
# claude-haiku-4-5:open-ai claude-sonnet-4-5:open-ai