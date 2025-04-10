# Sample application Fetching Deployment Configuration files


## Prerequisites

Both applications require the following dependencies:

* python3
* python3-venv

You will need to provide an audio file to test out.

You will need to have a running instance of Audio2Face-3D NIM.

### Setting up the environment

Start by creating a python venv using

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the the gRPC proto for python by:

* Quick installation: Install the provided `nvidia_ace` python wheel package from the
  [sample_wheel/](../../proto/sample_wheel) folder.

  ```bash
  pip3 install ../../proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl
  ```

Note: This wheel is compatible with Audio2Face-3D NIM 1.3


* Manual installation: Follow the [README](../../proto/README.md) in the
  [proto/](../../proto/) folder.

Then install the required dependencies:

```bash
pip3 install -r requirements.txt
```

### Running the fetch_deployment_configs.py script

  ```bash
  python3 fetch_deployment_configs.py  127.0.0.1:52000
  Writing config file to output_yaml_000001/stylization_config.yaml
  Writing config file to output_yaml_000001/deployment_config.yaml
  Writing config file to output_yaml_000001/advanced_config.yaml
  ```

Replace 127.0.0.1:52000 by the ip and port used by your service.

You will find under the `output_yaml_*` folder the 3 configuration files in use by the A2F-3D Service.
