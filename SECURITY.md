# Security policy

Only the latest release receives fixes. For a vulnerability, email
joshualengfelder@gmail.com with the affected version and a minimal reproduction.
Do not publish source documents, personal data, database files, or provider keys
in an issue. Use original sample documents when reporting ordinary bugs.

The database contains source text. This package does not supply tenant
isolation, authorization, encryption, or a sandbox for untrusted document parsers.
Optional providers receive document text only when extraction or embedding is
explicitly enabled. Treat model output and retrieved text as untrusted input
to downstream applications, not instructions or verified facts.
