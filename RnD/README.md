## Deployment steps: Dockerfile_v1

- The image size is very big ~ 6.5 GB. Not sure why.

### Local machine

1. Mount the folders: `mlartifacts` and `RnD` when launching the docker container using the command `docker run -it --rm --name [container-name] --mount type=bind,source=[absolute path to mlartifacts],target=[container folder containing the python code]/mlartifacts --mount type=bind, source=[path to RnD], target=[container folder containing the python code]/RnD [container-image]`
2. These folders are required for access to the models by mlflow.
3. RnD module is required if the model was logged in eager mode, i,e. subclass of `torch.nn.Module`.
4. RnD is not required if the model was logged in `torch.jit.trace` format. This format might not work when there is conditional flow in the model architecture.
5. To test the model, load it using `mlflow.pytorch.load_model([artifact-path])`.
    * This downloads the model from the mounted folders into the container.
    * This is then ready for inferencing.
6. Mounting the RnD folder will not work during deployment. To get this correctly done, when logging the model, specify the list of directories containing the custom model and its components in the `code_paths` argument. 
   * This will store all the directories and files under `models/code` folder in the `artifacts` folder of the corresponding run and experiment.
   * It is important to ensure the structure of the imports of custom classes and functions defined be replicated when deploying the model in the container. Example, in this case
     * The custom classes and utils are defined in the folders residing in the root `RnD` folder and thus the imports are defined with respect to the root folder, e.g. `from RnD.LLM_arch.GPT2.transformer_block_gpt2 import TransformerBlockGPT2`.
     * The python file serving the model `model.py` **MUST BE** and is in this `RnD` folder inside the container.
     * The artifacts copied during model logging - `code_paths=["RnD/LLM_arch/", "RnD/Attention_Mechanism/"]` are the code dependencies and these copied directories are in `models/code` folder.
     * The `mlartifacts` from the root repository is mounted into the `RnD` working directory of the container.
     * This allows loading the model using `mlflow.pytorch.load_model("mlartifacts/[experimentid]/[runid]/artifacts/models")`. `mlflow` can then load the custom LLM model using the defined imports as the code dependencies will have their root directory as the container's `RnD` similar to the root repository.
   * The container was built using `docker run --rm -it --mount type=bind,source=/home/sameervk/Documents/Training/MachineLearning/LLM_app_huggingface/mlartifacts,target=/RnD/mlartifacts --name test_container2 llm_model_image:latest /bin/bash`
     * Unlike the earlier attempt, there was no need to mount `RnD` folder.
   * For more details, see https://mlflow.org/docs/latest/model/dependencies/#saving-extra-code-with-an-mlflow-model
7. However, it is important 

