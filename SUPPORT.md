# Support

Need help with CortexNet? Use the channels below so maintainers can respond efficiently.

## 1. Bug Reports

Open a GitHub issue using the bug template:

- Include environment (`python`, `torch`, device backend)
- Include exact reproduction steps
- Include expected vs actual behavior
- Attach logs or traceback

## 2. Feature Requests

Open a GitHub issue using the feature template:

- Describe use case and expected value
- Describe API or UX impact
- If possible, include a minimal proposal

## 3. Security Issues

Do not post security vulnerabilities publicly.

Follow:

- `SECURITY.md`

## 4. Response Expectations

Best-effort response targets:

- New bug triage: within 72 hours
- Security acknowledgement: within 24 hours
- Feature discussion: within 7 days

## 5. Before Opening an Issue

Please run these checks first:

```bash
make lint
make test-all
make check
```

Also verify you are on the latest release from:

- https://pypi.org/project/cortexnet/
