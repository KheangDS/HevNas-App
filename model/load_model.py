import torch
from model.model import KlebJeb

def load_trained_model(model_path: str, num_classes: int, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = KlebJeb(input_shape=3, hidden_units=64, output_shape=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model
