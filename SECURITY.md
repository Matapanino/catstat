# Security Policy

## Supported versions

`catstat` is pre-1.0; security fixes target the latest released version on PyPI.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

Please report security issues **privately** — not in public issues or pull requests:

- Preferred: open a private advisory via
  [GitHub Security Advisories](https://github.com/Matapanino/catstat/security/advisories/new).
- Or email the maintainer: mkawamata038@gmail.com.

Include a description, reproduction steps, and the affected version. We aim to acknowledge within a
few days and to coordinate a fix and disclosure timeline with you.

## Scope note

`catstat` processes user-provided data with pandas/numpy (and optionally cuDF/CuPy) and executes no
untrusted code on its own. However, **custom aggregation callables** passed to `stats=` run as-is —
only pass callables you trust.
