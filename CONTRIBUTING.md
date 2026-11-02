# Contributing

Issues and focused pull requests are welcome.

## Before opening a pull request

```sh
python -m pip install -e '.[dev]'
python -m pytest
```

Keep the default path offline: tests and the demo must not require an API key
or make paid calls. New retrieval or extraction behavior needs a test that
shows the evidence it returns.

## Scope

The engine is intentionally small: SQLite, FTS5, chunk/evidence IDs, optional
provider adapters. Avoid turning it into a hosted service, an answer generator,
or a vector database. Those can be separate layers.

Do not commit private corpora, credentials, or provider keys. Sample data must
be original or clearly reusable.

## Reporting bugs

Include the Python version, SQLite version, a minimal document or query that
reproduces the issue, and the expected versus actual evidence.
