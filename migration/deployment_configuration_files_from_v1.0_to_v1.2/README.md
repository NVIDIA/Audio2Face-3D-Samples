# Config Migration

This sample python app allows you to migrate your A2F-3D config files from v1.0 to v1.2.

Note: Audio2Face-3D NIM 1.3 is backward compatible with v1.2 configs.

## Prerequisites

Install:

* python3
* python3-venv

Set up a virtual environment and install the needed packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Migrating configs

To do so there are 2 possibilities.
You want to migrate the A2F-3D config files used for:

1. running the docker container
2. deploying the UCS app

### Updating docker container configs

Update these files with your own config file:

* [docker_container_configs/a2f_config.yaml](docker_container_configs/a2f_config.yaml)
* [docker_container_configs/ac_a2f_config.yaml](docker_container_configs/ac_a2f_config.yaml)

Then run:

```bash
python3 convert_configuration_files.py docker_config
```

This will create new configuration files compatible with A2F-3D v1.2 and display the folder name where they are located.

### Updating the UCS app configs

Update [ucs_app_configs/a2f_config.yaml](ucs_app_configs/a2f_config.yaml) with your own config file.

Then run:

```bash
python3 convert_configuration_files.py ucs
```

This will create new configuration files compatible with A2F-3D v1.2 and display the folder name where they are located.
