# Python proto services

You can either:

* Install the provided `nvidia_ace` python wheel package from the
[sample_wheel/](./sample_wheel/) folder with pip:

Note: This wheel is compatible with Audio2Face-3D NIM 1.3

```bash
pip3 install ./sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl
```

* Generate the gRPC python module and manually copy it in the same folder as the
script you want to run.
* Generate the gRPC python module and make your own python wheel package.

**Note:** Audio2Face-3D NIM v1.3 continues to use v1.2.0 of the nvidia_ace gRPC python module.

## Generate the gRPC python module

Start by creating a python venv using

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the python `requirements.txt`, then generate the python module with
`generate_code_from_protos.py` script:

```bash
pip3 install -r requirements.txt
python3 generate_code_from_protos.py files_to_compile.yaml
```

This will make a new folder with the generated protos in
`generated/nvidia_ace/`.

## Make a python wheel

### Requirements

* Generate the python module.
* Install `setuptools` and `wheel` python package.

Start by creating a python venv using

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install `setuptools` and `wheel` dependencies.

```bash
pip3 install setuptools
pip3 install wheel
```

### Usage

Move the files in the `generated/` folder here by running:

```bash
mv generated/* .
```

The final structure should be like:

```bash
setup.py
nvidia-ace/
├── __init__.py
└── ... (other Python files)
```

Make a wheel by running:

```bash
python3 setup.py bdist_wheel
```

You can find the wheel in `dist` folder.
