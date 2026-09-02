# Benchmark Results

### Benchmark Results (K=5)

| Metric | Baseline (FAISS-only) | Hybrid (FAISS+FTS+RRF) | Δ |
|:---|---:|---:|---:|
| Recall@5 | 0.6667 | 0.7333 | +10.0% |
| Precision@5 | 0.2267 | 0.3200 | +41.2% |
| MRR | 0.3822 | 0.6333 | +65.7% |
| Avg latency (ms) | 32.0784 | 1522.4843 | +1490ms |

### Per-Query Detail

| Query | Approach | Recall@5 | Precision@5 | MRR | Latency (ms) |
|:---|:---|---:|---:|---:|---:|
| Сколько длится проект Ecto-1 и сколько человек ... | baseline | 1.00 | 0.20 | 0.25 | 88 |
| Сколько длится проект Ecto-1 и сколько человек ... | hybrid | 1.00 | 0.40 | 1.00 | 1882 |
| Кто отвечает за документацию в команде? | baseline | 0.00 | 0.00 | 0.00 | 8 |
| Кто отвечает за документацию в команде? | hybrid | 0.00 | 0.00 | 0.00 | 1062 |
| Какие церемонии входят в командный процесс? | baseline | 1.00 | 0.20 | 0.20 | 8 |
| Какие церемонии входят в командный процесс? | hybrid | 1.00 | 0.20 | 1.00 | 3824 |
| Какие защищённые маршруты есть в приложении? | baseline | 0.00 | 0.00 | 0.00 | 7 |
| Какие защищённые маршруты есть в приложении? | hybrid | 0.00 | 0.00 | 0.00 | 976 |
| Как работает авторизация через SSO? | baseline | 1.00 | 0.20 | 0.25 | 9 |
| Как работает авторизация через SSO? | hybrid | 1.00 | 0.60 | 1.00 | 1018 |
| slowapi rate limiting 429 | baseline | 1.00 | 0.40 | 0.50 | 63 |
| slowapi rate limiting 429 | hybrid | 1.00 | 0.40 | 1.00 | 1435 |
| AES-256-CBC HMAC токен | baseline | 0.00 | 0.00 | 0.00 | 50 |
| AES-256-CBC HMAC токен | hybrid | 1.00 | 0.40 | 0.50 | 1545 |
| pgadmin 5050 postgres 5433 | baseline | 1.00 | 0.60 | 1.00 | 50 |
| pgadmin 5050 postgres 5433 | hybrid | 1.00 | 0.60 | 1.00 | 1732 |
| squash merge two approvals | baseline | 1.00 | 0.40 | 1.00 | 135 |
| squash merge two approvals | hybrid | 1.00 | 0.40 | 1.00 | 1379 |
| VITE_APP_ENV VITE_BYPASS_AUTH | baseline | 1.00 | 0.20 | 0.20 | 18 |
| VITE_APP_ENV VITE_BYPASS_AUTH | hybrid | 1.00 | 0.40 | 0.50 | 971 |
| Какой port у pgadmin в docker compose? | baseline | 1.00 | 0.60 | 1.00 | 8 |
| Какой port у pgadmin в docker compose? | hybrid | 1.00 | 0.60 | 0.50 | 904 |
| Какой email и пароль для Render.com? | baseline | 1.00 | 0.20 | 1.00 | 8 |
| Какой email и пароль для Render.com? | hybrid | 1.00 | 0.20 | 1.00 | 932 |
| Какие миграции Alembic существуют? | baseline | 0.00 | 0.00 | 0.00 | 13 |
| Какие миграции Alembic существуют? | hybrid | 0.00 | 0.00 | 0.00 | 1182 |
| Существует ли роут для отдельной статьи? | baseline | 0.00 | 0.00 | 0.00 | 6 |
| Существует ли роут для отдельной статьи? | hybrid | 0.00 | 0.00 | 0.00 | 1180 |
| В течение скольких дней можно восстановить удал... | baseline | 1.00 | 0.40 | 0.33 | 9 |
| В течение скольких дней можно восстановить удал... | hybrid | 1.00 | 0.60 | 1.00 | 2813 |
