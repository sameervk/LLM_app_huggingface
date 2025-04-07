# LLM_app_huggingface
FastAPI app using LLMs deployed on HuggingFace Spaces


## LLMs

### 1. GPT2:
   * Developed from scratch: path - RnD/LLM_arch/GPT2
   * Generates text given a prompt 


## Dependencies

* The virtual environments for LLM architecture development (RnD folder) and FastAPI app (api) have been separately setup using `uv`.
  * Although it appears more or less all the libraries being used in `RnD` are required for the `api`.
  * So one `venv` in the root folder should suffice.
  * Similar one `Dockerfile` in the root folder should suffice to deploy the app.
* The process below *did not work* because in the api's `main.py`, when loading model, `mlflow` attempts to install it's own environment. However the code has been setup to run in its own `uv venv` environment. This process could work when deploying the model as an **endpoint.**
  * Before running the `mlflow` *experiments*, ensure the custom LLM architecture code with all the required classes used in the experiments are **packaged in .whl** format. This is required when logging a pytorch model using `mlflow` using the `extra_pip_requirements` and `code_paths` arguments.
    * For more info on why this argument must be set and the structure of custom code see the `README.md` file of `RnD`.
      *  For more information about logging `mlflow` models with custom code dependencies, see: https://mlflow.org/docs/latest/model/dependencies/#how-mlflow-records-dependencies
    * The package build configuration is set in the root `pyproject.toml` file. 
      * `setuptools`/`hatchling` can be used to build the package.
      * `setup.py` will be deprecated in the future.
      * The command to execute is `python -m build --skip-dependency-check --wheel`.
  * Instead, push the `.whl` to github repo and in the `api`'s `pyproject.toml`, add this link to `uv.tool.sources` and `uv` should install it along with other packages.
    * To install a specific `.whl` blob from a personal or private github repo, an access token is required. Once this is obtained, use the following code to install it in the virtual environment.
      * `pip`: `pip install https://{ACCESS_TOKEN}@raw.githubusercontent.com/{username}/{repo}/{branch}/{path}/{to}/{.whl}`
      * `uv`: `uv add https://{ACCESS_TOKEN}@raw.githubusercontent.com/{username}/{repo}/{branch}/{path}/{to}/{.whl}`
    * This implies that the above `extra_pip_requirements` and `code_paths` arguments are not required when logging the model.
* To pass an access token during docker build, **DO NOT** pass the secrets as ARG or ENV. Instead pass it using the secrets mount option.
  * In the Dockerfile, `RUN --mount=type=secret,id=llm_arch,env=LLM_ARCH_DOWNLOAD_ACCESS_TOKEN\
    uv add https://{llm_arch}@raw.githubusercontent.com/sameervk/LLM_app_huggingface/main/dist/llmarch-0.0.1-py3-none-any.whl`
  * Export the environment variables: `source var.env`
  * In the build command, `docker build --secret id=llm_arch,env=$LLM_ARCH_DOWNLOAD_ACCESS_TOKEN -t [image-name:tag] -f Dockerfile . `
  * For more information about passing secrets, see https://docs.docker.com/build/building/secrets/
  * For secrets with docker compose: https://docs.docker.com/compose/how-tos/use-secrets/
* See `README` of `api` folder to start the container after building the image using a Dockerfile.

## Docker Compose

### Development
#### `watch` mode
* The config is in `compose_api_watch.yaml`.
* This is to quickly test the application execution in a docker container given any changes to the code.
* Run `docker compose -f compose_api_watch.yaml up api --watch`
  * Need the `--watch` option, else it is not enabled.
* It a directory is to be `watched`, do not specify it as a `volume` too. Else, `volume` takes precedence.

#### using `volume`
* For developing in a container.
* The command that needs to be set in the compose file to start the container is `sleep infinity`.
* Interestingly, when `api` folder is specified as volume, for some reason, the `RnD .whl` package is **_missing_** even after installation during image build.
  * _Not sure what is happening here._
* Run `docker compose -f compose_api_volume.yaml up -d api` to build and start the container.

### Production
* Use the similar config as `Development: volume`
