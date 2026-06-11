# CONTRIBUTING.md: Contribution Guidelines

We welcome contributions to the LISA ecosystem. Please follow this guide to set up your environment, write tests, and submit PRs.

---

## 💻 Development Environment Setup

1. Fork and clone the repository.
2. Initialize and activate the virtual environment.
3. Install development tools:
   ```bash
   pip install -r requirements.txt
   pip install pytest black ruff
   ```
4. Set up pre-commit hooks to automatically format files:
   ```bash
   ruff check --fix .
   black .
   ```

---

## 🧪 Testing Guidelines

* **Unit Tests**: Place in the `tests/` directory and run via:
  ```bash
  python -m pytest tests/
  ```
* **Coverage**: Ensure code coverage is maintained.
* **Async Code**: Use `asyncio.run()` inside synchronous test wrapper functions to prevent plugin setup issues.

---

## 📨 Pull Request Checklist
Before opening a PR, ensure:
1. All linting checks pass without warnings.
2. Every test runs successfully.
3. Your commit messages follow structured formatting: `feat: add custom tool registry` or `fix: handle db lock error`.
4. The README or documentation is updated to match your changes.
