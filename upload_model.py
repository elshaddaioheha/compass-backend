import os

from huggingface_hub import HfApi, login


token = os.getenv("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN environment variable is required to upload the model.")

login(token=token)

api = HfApi()

print("Uploading ONNX model files...")
api.upload_folder(
    folder_path='./onnx_model',
    repo_id='Oheha/compass-emotion-classifier',
    repo_type='model',
    path_in_repo='onnx_model',   # saved as onnx_model/ inside the HF repo
)

print("Upload complete! Model is live at: https://huggingface.co/Oheha/compass-emotion-classifier")
