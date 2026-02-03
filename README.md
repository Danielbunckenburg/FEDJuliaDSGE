# FED DSGE Model (Python/C Implementation)

This project is a high-performance Python and C implementation of the NY Fed's DSGE models (specifically Model 1002), originally ported from `DSGE.jl`.

## Project Structure

- `src/python/`: Core logic and model definitions.
- `src/c/`: Performance-critical kernels for solving and filtering (Gensys, Kalman).
- `data/`: Input data (FRED observables, samples).
- `scripts/`: Utility scripts for compilation and execution.
- `tests/`: Unit and integration tests.
- `outputs/`: Generated results, logs, and figures.
- `docs/`: Additional documentation.

## Getting Started

### 1. Prerequisites
- Python 3.9+
- A C compiler (Clang or GCC recommended). See `docs/INSTRUCTIONS_C_COMPILER.md`.

### 2. Compile C Extensions
Run the compilation script from the root directory:
```powershell
python scripts/compile.py
```

### 3. Run the Model
You can run the main pipeline to verify everything is working:
```powershell
python run_pipeline.py
```

To run with real FRED data (requires an API key):
```powershell
$env:FRED_API_KEY = "your_key"
python scripts/run_fred_estimation.py
```

## Testing
Run all tests using `pytest`:
```bash
pytest tests/
```
