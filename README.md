# Python Research

A lightweight research playground for experimenting with AI behavior, logging, and analysis.
This project is meant to be run locally and explored, not installed as a library.

---

## Project Structure

```
python-Research/
├── ai/
│   ├── __init__.py
│   └── logger.py        # Core logging / memory utilities
├── .gitignore
├── analyzer.py          # Analysis + visualization logic
├── main.py              # Entry point for running experiments
├── requirements.txt     # Python dependencies
└── README.md
```

---

## How to Use

### 1. Clone the repository

```
git clone https://github.com/Smarg1/python-Research.git
cd python-Research
```

### 2. Create a virtual environment

```
python -m venv .venv
```

### 3. Activate the environment

**macOS / Linux**

```
source .venv/bin/activate
```

**Windows (PowerShell)**

```
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the project

```
python main.py
```

---

## Observations & Analysis

* Observations are stored in the `memories` directory.
* Intelligent forgetting logs are stored in the `forget` directory.
* To visualize and analyze results, run:

  ```
  python analyzer.py
  ```

  This opens a windowed display with graphs and research metrics.
* For raw logs, inspect the `memories` and `forget` folders directly.

---

## Notes

* This project assumes a local, experimental workflow.
* Files and directories may evolve as research progresses.
* Structure exists to support iteration, not permanence.
