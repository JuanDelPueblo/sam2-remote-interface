# SAM 2 Universal Web API

Simple web API for SAM 2. Currently only supports images but support for videos is coming soon.

## How to use

1. Create Python virtual environment using venv or Anaconda

2. Run `git clone https://github.com/facebookresearch/sam2.git && cd sam2` outside of this directory

3. Run `pip install -e .` while in `sam2` directory

4. Run `pip install -e ".[notebooks]"` while in `sam2` directory

5. Return to this project's directory and run

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

6. Run `fastapi dev api.py` to test the API manually with hot reload, or run `python3 tester.py` to test the API automatically and see results under `images/`