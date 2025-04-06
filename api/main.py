import os
import dotenv
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
import mlflow


env_var = dotenv.load_dotenv("app_config.env")

if env_var:
    pass
else:
    raise FileNotFoundError("Application environment variables file not found.")

app_title="Generative AI app"
app_summary = "App responding to prompts. Uses GPT2 architecture"
app = FastAPI(title=app_title,
              summary=app_summary)

origins = os.getenv("ALLOWED_ORIGINS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# download model
# mlflow.set_tracking_uri("http://localhost:8080")
# model_uri = "../mlartifacts/368316252121826279/70d9c885e3e84885b91585640a06084f/artifacts/models"
model_uri = os.getenv("MODEL_URI")

model=mlflow.pytorch.load_model(model_uri, map_location="cpu")
print(model)

@app.get(path="/", status_code=status.HTTP_200_OK)
async def home():

    return {"message": f"Hello. Welcome to {app_title}. {app_summary}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app="main:app",
                host="localhost",
                port=5000,
                reload=True,
                )
