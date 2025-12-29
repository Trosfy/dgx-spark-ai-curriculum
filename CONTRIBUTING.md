# Contributing to DGX Spark AI Curriculum

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Style Guidelines](#style-guidelines)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

This project follows a simple code of conduct:

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Be patient** with learners
- **Focus on what's best** for the community

---

## How Can I Contribute?

### 🐛 Reporting Bugs

Found a bug in the curriculum materials, code, or documentation?

1. Check if the issue already exists in [GitHub Issues](https://github.com/YOUR_USERNAME/dgx-spark-ai-curriculum/issues)
2. If not, [create a new bug report](https://github.com/YOUR_USERNAME/dgx-spark-ai-curriculum/issues/new?template=bug_report.md)
3. Include:
   - Module and task location
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (DGX OS version, container, etc.)

### 💡 Suggesting Enhancements

Have an idea for improvement?

1. [Create a feature request](https://github.com/YOUR_USERNAME/dgx-spark-ai-curriculum/issues/new?template=feature_request.md)
2. Describe the problem and proposed solution
3. Explain how it benefits DGX Spark users

### 📝 Improving Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples or clarifications
- Improve setup instructions
- Update outdated information

### 🔧 Contributing Code

Code contributions can include:

- New utility scripts
- Bug fixes in existing code
- New notebook tasks
- Performance optimizations

### 📚 Adding Curriculum Content

Want to add new learning materials?

- New notebooks for existing modules
- Additional tasks or exercises
- Solution notebooks
- Supplementary guides

---

## Getting Started

### Prerequisites

- NVIDIA DGX Spark (or ability to test on compatible hardware)
- Git
- Python 3.10+
- Familiarity with the curriculum content

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/dgx-spark-ai-curriculum.git
cd dgx-spark-ai-curriculum
git remote add upstream https://github.com/ORIGINAL_OWNER/dgx-spark-ai-curriculum.git
```

### Set Up Development Environment

```bash
# Create virtual environment (or use NGC container)
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install black isort flake8 pytest nbformat
```

---

## Development Workflow

### 1. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Follow the [style guidelines](#style-guidelines)
- Test your changes on DGX Spark if possible
- Update documentation as needed

### 3. Validate Changes

```bash
# Format Python code
black utils/
isort utils/

# Lint
flake8 utils/ --max-line-length=100

# Validate notebooks
python -c "
import nbformat
from pathlib import Path
for nb in Path('.').rglob('*.ipynb'):
    if '.ipynb_checkpoints' not in str(nb):
        nbformat.read(open(nb), as_version=4)
        print(f'✓ {nb}')
"
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "type: brief description

Longer explanation if needed.

Fixes #123"
```

**Commit types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## Style Guidelines

### Python Code

- **Formatter:** Black (line length 100)
- **Import sorting:** isort
- **Linting:** flake8
- **Docstrings:** Google style

```python
def function_name(param1: str, param2: int = 10) -> dict:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> result = function_name("test", 20)
    """
    pass
```

### Jupyter Notebooks

- Clear all outputs before committing (or keep minimal outputs)
- Include markdown cells explaining each section
- Use the template from `templates/notebook_template.ipynb`
- Add learning objectives at the top
- Include cleanup cells at the bottom

### Markdown

- Use ATX-style headers (`#`, `##`, `###`)
- One sentence per line (for better diffs)
- Use fenced code blocks with language identifiers
- Include alt text for images

### Module READMEs

Follow the template in `templates/module_readme_template.md`:

- Learning outcomes (what students will achieve)
- Learning objectives (specific, measurable, with Bloom's taxonomy)
- Tasks with time estimates and deliverables
- DGX Spark-specific guidance
- Milestone checklist

---

## Submitting Changes

### Pull Request Process

1. **Create PR** against the `main` branch
2. **Fill out the PR template** completely
3. **Wait for review** - maintainers will review within a few days
4. **Address feedback** - make requested changes
5. **Merge** - once approved, your PR will be merged

### PR Requirements

- [ ] Code follows style guidelines
- [ ] All notebooks are valid JSON
- [ ] Documentation is updated
- [ ] Tested on DGX Spark (if applicable)
- [ ] No large files or outputs committed

### Review Criteria

PRs are evaluated on:

- **Correctness:** Does it work as intended?
- **Clarity:** Is the code/content easy to understand?
- **DGX Spark relevance:** Does it leverage or document DGX Spark capabilities?
- **Educational value:** Does it help learners?

---

## Recognition

Contributors are recognized in:

- GitHub contributors list
- Release notes for significant contributions
- Acknowledgments in affected modules

---

## Questions?

- Open a [question issue](https://github.com/YOUR_USERNAME/dgx-spark-ai-curriculum/issues/new?template=question.md)
- Check existing issues and discussions

---

Thank you for contributing to the DGX Spark AI Curriculum! 🚀
