import mlflow
from pathlib import Path
import dotenv
import os

import torch.cuda
from tokenizers import Tokenizer

from RnD.Training_and_Evaluation.pytorch.utils import generate_text

if __name__ == "__main__":
    dotenv.load_dotenv("RnD/Training_and_Evaluation/mlflow_pytorch/mlflow_config.env")

    uri_host = os.getenv("HOST") + ":" + os.getenv("PORT")
    print(f"Tracking server: {uri_host}")

    # set tracking uri
    mlflow.set_tracking_uri(uri=uri_host)
    # model_uri = "mlflow-artifacts:/368316252121826279/05a527889cdf403683415e0e54adebdc/artifacts/models"
    model_uri = "models:/custom_gpt2_model@production"

    loaded_model = mlflow.pytorch.load_model(model_uri)

    test_prompt = "What is "
    tokenizer = Tokenizer.from_pretrained("gpt2")
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded_model.to(compute_device)

    generated_text = generate_text(
        model=loaded_model,
        prompt=test_prompt,
        num_output_tokens=20,
        tokenizer=tokenizer,
        context_len=256,
        compute_device=compute_device,
    )
    print(
        f"\nGenerated text before training given prompt '{test_prompt}' : {generated_text}"
    )
