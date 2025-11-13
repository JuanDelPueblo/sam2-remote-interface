# Data Engine

A complete data annotation and tracking system with both Python (FastAPI) backend and C++ interface. Supports video object segmentation using SAM 2 and point tracking using CoTracker.

## Using and testing backend

1. Create Python virtual environment using venv or Anaconda

2. Run `git clone https://github.com/facebookresearch/sam2.git && cd sam2` outside of this directory

3. Run `pip install -e .` while in `sam2` directory

4. Run `pip install -e ".[notebooks]"` while in `sam2` directory

5. Return to the data engine directory and run `pip install -r requirements.txt`

6. Run `fastapi dev backend/api.py` to test the API manually with hot reload, or run `python3 backend/tests/tester.py` to test the API automatically and see results under `backend/tests/`
