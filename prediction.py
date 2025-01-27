from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image

def pred_class(model: torch.nn.Module, image: Image.Image, class_names: List[str], image_size: Tuple[int, int] = (224, 224)):
    # Get device from model
    device = next(model.parameters()).device
    
    # Create transformation pipeline
    image_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transform image and move to correct device
    transformed_image = image_transform(image).unsqueeze(dim=0)
    transformed_image = transformed_image.to(device)

    # Prediction
    model.eval()
    with torch.no_grad():
        target_image_pred = model(transformed_image)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    
    return target_image_pred_probs.cpu().numpy()