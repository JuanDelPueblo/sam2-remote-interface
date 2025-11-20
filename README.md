# Data Engine

A complete data annotation and tracking system with both Python (FastAPI) backend and C++ interface. Supports video object segmentation using SAM 2 and point tracking using CoTracker.

## Install Dependencies

1. Create Python virtual environment using venv or Anaconda

2. Run `git clone https://github.com/facebookresearch/sam2.git && cd sam2` outside of this directory

3. Run `pip install -e .` while in `sam2` directory

4. Run `pip install -e ".[notebooks]"` while in `sam2` directory

5. Return to the data engine directory and run `pip install -r requirements.txt`

## Testing via Python

1. Run `fastapi dev backend/api.py` to test the API manually with hot reload, or run `python3 backend/tests/tester.py` to test the API automatically and see results under `backend/tests/`

## Testing via C++

1. Install `nlohmann_json` and `curl` dev packages via your package manager

2. Run `mkdir build`

3. Run `cmake build` to generate CMake build files

4. Run `cmake --build build` to compile project

5. Run `./build/src/BackendInterfaceTests` while the venv is sourced to test backend code