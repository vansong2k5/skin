# Contributing to Skin Insight

We are thrilled that you want to contribute! This document outlines how to work
with the repository and submit high-quality changes.

## Code of Conduct

All contributors must follow the [Contributor Covenant](CODE_OF_CONDUCT.md).
Please report unacceptable behaviour to the maintainers listed in
[SECURITY.md](SECURITY.md).

## Development workflow

1. **Fork the repository** and create a feature branch (`git checkout -b
   feature/add-awesome-thing`).
2. **Create a virtual environment** and install dependencies with `pip install
   -r requirements.txt`.
3. **Run the checks** before sending a PR:
   - `python -m compileall backend`
   - Add unit or integration tests when possible.
4. **Write descriptive commits and PRs**. Explain *why* the change is needed and
   reference related issues.
5. **Submit a pull request** targeting the `main` branch. Fill in the PR
   template (or provide equivalent information) covering motivation, testing and
   risk assessment.

## Style guidelines

- Follow PEP 8 (4 spaces, descriptive variable names, type hints).
- Keep functions small and focused. If a file exceeds ~400 lines consider
  splitting it into modules.
- Document public modules, functions and classes using docstrings.
- Avoid hard-coding secrets. Use environment variables and document them in the
  README.

## Adding dependencies

- Prefer lightweight, well-maintained packages. Justify new dependencies in your
  PR description.
- Update `requirements.txt` and, if the dependency is optional, document how to
  enable it in the README.

## Reporting bugs

- Open an issue with steps to reproduce, expected vs. actual behaviour and, when
  possible, screenshots or sample payloads.
- Security-related issues should be reported privately following
  [SECURITY.md](SECURITY.md).

Thank you for helping build a trustworthy dermatology assistant!
