# Data Engine

Currently only the backend API for SAM 2 and cotracker is implemented to interact with C++ frontend. Supports both images and video masking.

## How to use

1. Create Python virtual environment using venv or Anaconda

2. Run `git clone https://github.com/facebookresearch/sam2.git && cd sam2` outside of this directory

3. Run `pip install -e .` while in `sam2` directory

4. Run `pip install -e ".[notebooks]"` while in `sam2` directory

5. Return to the data engine directory and run `pip install -r requirements.txt`

6. Run `fastapi dev api.py` to test the API manually with hot reload, or run `python3 tester.py` to test the API automatically and see results under `images/`
