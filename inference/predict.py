import torch

def predict_image(model, image_tensor, class_names):
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    return class_names[pred_idx], probs[0][pred_idx].item()
