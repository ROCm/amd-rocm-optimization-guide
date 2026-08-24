# AMD ROCm Optimization Guide

The AMD ROCm Optimization Guide focuses on performance optimization techniques for AMD GPUs using the HIP programming language. This repository provides comprehensive tutorials and best practices for maximizing GPU performance, covering essential topics such as parallel workload optimization, reduction operations, memory coalescing, and multi-GPU programming.

The purpose of this repository is to:

- Provide focused documentation on ROCm performance optimization techniques.

- Deliver comprehensive tutorials for optimizing HIP applications on AMD GPUs.

- Serve as a resource for developers seeking to maximize GPU performance.

## Build the documentation

You can build our documentation via the command line using Python.

See the `build.tools.python` setting in the [Read the Docs configuration file](https://github.com/ROCm/ROCm/blob/develop/.readthedocs.yaml) for the Python version used by Read the Docs to build documentation.

See the [Python requirements file](https://github.com/ROCm/ROCm/blob/develop/docs/sphinx/requirements.txt) for Python packages needed to build the documentation.

Use the Python Virtual Environment (`venv`) and run the following commands from the project root:

### Linux and WSL

```sh
python3 -mvenv .venv

.venv/bin/python -m pip install -r docs/sphinx/requirements.txt
.venv/bin/python -m sphinx -T -E -b html -d _build/doctrees -D language=en docs _build/html
```

### Windows

```powershell
python -mvenv .venv

.venv\Scripts\python.exe -m pip install -r docs/sphinx/requirements.txt
.venv\Scripts\python.exe -m sphinx -T -E -b html -d _build/doctrees -D language=en docs _build/html
```

Navigate to `_build/html/index.html` and open this file in a web browser.

For further information, please check [building documentation](https://rocm.docs.amd.com/en/latest/contribute/building.html).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development
workflow, issue tracking, and pull request guidelines.

## Security

To report a security vulnerability, see [SECURITY.md](SECURITY.md).
