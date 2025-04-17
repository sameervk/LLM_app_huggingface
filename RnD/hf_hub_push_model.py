import torch
from huggingface_hub import PyTorchModelHubMixin, HfApi
import dotenv
import os
import yaml
import mlflow

from RnD.LLM_arch.GPT2.llm_gpt2 import GPT2
from RnD.Training_and_Evaluation.pytorch.utils import generate_text

env_var = dotenv.load_dotenv(".env")
if os.getenv("GITHUB_ACTION_ACCESS_TOKEN"):
    pass
else:
    raise FileNotFoundError("No HF access token")


class GPT2HF(
    GPT2,
    PyTorchModelHubMixin,
    # metadata
    repo_url=os.getenv("REPO_URL"),
    license="gpl-3.0",
):
    """
    Extending the parent to enable upload to HF Hub
    """

    def __init__(self, **cfg):
        super().__init__(**cfg)


if __name__ == "__main__":
    username = os.getenv("HF_SPACE_USERNAME")
    hf_space_name = os.getenv("HF_SPACE_REPONAME")
    mlflow_model_uri = os.getenv("MLFLOW_MODEL_URI")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    # trained_model = mlflow.pytorch.load_model(str(Path.cwd().parent.joinpath(mlflow_model_uri)))
    trained_model = mlflow.pytorch.load_model(mlflow_model_uri)
    # artifacts = mlflow.artifacts.download_artifacts(artifact_uri=mlflow_model_uri,
    #                                                 tracking_uri=mlflow_tracking_uri
    #                                                 )
    # This only downloads the artifacts in the models folder. To download the artifacts from the higher runid folder, need runid

    # ------DOWNLOAD LLM CONFIG OF THE REGISTERED MODEL--------
    mlflow_client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)
    model_version = mlflow_client.get_model_version_by_alias(
        name="custom_gpt2_model", alias="production"
    )
    path_to_config = mlflow_client.download_artifacts(
        run_id=model_version.run_id, path="llm_config.yaml"
    )
    # path_to_config = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{model_version.run_id}/llm_config.yaml")

    with open(path_to_config, "r") as file:
        llm_config = yaml.safe_load(file)
        print(llm_config)

    model_to_hf_hub = GPT2HF(**llm_config)
    # copy weights from trained
    model_to_hf_hub.load_state_dict(trained_model.state_dict())

    # NOT working
    # load weights
    # path_to_weights = Path.cwd().parent.joinpath(model_uri).joinpath("data/model.pth")
    # with torch.serialization.safe_globals([GPT2]):
    # model_to_hf_hub.load_state_dict(torch.load(str(path_to_weights), weights_only=True))

    # ------------------------------
    # ------PUSH TO HF HUB---------
    model_to_hf_hub.push_to_hub(
        username + "/" + model_version.name,
        config=llm_config,
        token=os.getenv("GITHUB_ACTION_ACCESS_TOKEN"),
    )

    # --------------------------------------------
    # -------Test download model: WORKS-----------
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained("gpt2")
    compute_device = torch.device("cpu")
    new_model = GPT2HF(**llm_config)
    load_test_model = new_model.from_pretrained(
        username + "/" + model_version.name,
        token=os.getenv("GITHUB_ACTION_ACCESS_TOKEN"),
        # config=llm_config
    )
    # load_test_model = AutoModel.from_pretrained(
    #     username + "/" + model_version.name,
    #     token=os.getenv("GITHUB_ACTION_ACCESS_TOKEN"),
    #     config=llm_config
    # )
    # # THIS DOES NOT WORK. A TypeError occurs

    print(load_test_model)
    load_test_model.eval()

    test_prompt = "What is"

    generated_text = generate_text(
        model=load_test_model,
        prompt=test_prompt,
        num_output_tokens=20,
        tokenizer=tokenizer,
        context_len=256,
        compute_device=compute_device,
    )
    print(
        f"\nGenerated text before training given prompt '{test_prompt}' : {generated_text}"
    )

    # -----------------------------------
    # ---------RESTART HF SPACE----------
    hf_api = HfApi()
    try:
        space_run_time_info = hf_api.restart_space(
            repo_id=username + "/" + hf_space_name,
            token=os.getenv("GITHUB_ACTION_ACCESS_TOKEN"),
        )
    except Exception as err:
        print(err)
