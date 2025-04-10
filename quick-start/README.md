# Audio2Face-3D NIM

## Prerequisite

Make sure:

* [Docker](https://docs.docker.com/get-docker/)
* [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* [Docker Compose](https://docs.docker.com/compose/install/)

are installed on your machine.

Set your NGC API key in the `NGC_API_KEY` environment variable.

```bash
export NGC_API_KEY=<your_ngc_api_key>
```

## Getting started

To run the containers on your machine you need to run docker compose in the same directory as the  `docker-compose.yml` file.

For `claire` model:

```bash
A2F_3D_MODEL_NAME=claire docker compose up
```

For `mark` model:

```bash
A2F_3D_MODEL_NAME=mark docker compose up
```

For `james` model:

```bash
A2F_3D_MODEL_NAME=james docker compose up
```

The first run will take several minutes as both Audio2Emotion and Audio2Face-3D TRT model have to be generated.
For the subsequent starts, this step will be cached.

You can Ctrl+C to interrupt this deployment.

You will see some logs containing `Running...` in stdout when Audio2Face-3D is up and running.

## Model cache

The Audio2Face-3D will cache the generated TRT engines in `a2f-3d-init-data` folder. To invalidate the cache, delete the `*.trt` files from inside the directory.

## Configuration files

The provided configuration file will be used by Audio2Face-3D at Startup.
If you update them, you will need to interrupt and run again the containers.
(Ctrl+C and then `$ docker compose up`)

## Observability

We provide a separate `docker-compose-with-observability.yml` file to quick start metrics and traces collection and visualization.
You can start it by running the following command and replacing `model_name` with either `claire`, `mark` or `james`:

**Keep in mind** that [configs/deployment_config.yaml](../configs/deployment_config.yaml) file mounted inside the container will be changed,
and the `telemetry.metrics_enabled` and `telemetry.traces_enabled` options will be `True`.

```bash
A2F_3D_MODEL_NAME={model_name} docker compose -f docker-compose-with-observability.yml up
```

To visualize the metrics go to:

* [http://localhost:9090/graph](http://localhost:9090/graph) for Prometheus UI.
* [http://localhost:16686/search](http://localhost:16686/search) for Jaeger UI.

## Troubleshooting

If your deployment is in an invalid state, or encounters errors while starting, you can clean up local running dockers.

To remove the cached Audio2Emotion and Audio2Face-3D models:

```bash
A2F_3D_MODEL_NAME={model_name_used_at_startup} docker compose down -v
```

To remove all current docker containers:

```bash
A2F_3D_MODEL_NAME={model_name_used_at_startup} docker container prune -f 
```
