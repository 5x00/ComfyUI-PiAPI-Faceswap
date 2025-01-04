import PIL
import numpy as np
import base64
import requests
import json
import PIL
from io import BytesIO
import torch
import time

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def fetch_image(url, payload, headers, timeout=3, interval=0.5):
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Make the GET request
        response = requests.request("GET", url, headers=headers, json=payload)
        
        if response.code == 200:
            data = response.json()
            
            # Check if the response has the expected data (e.g., status is not "pending")
            if data.get("data", {}).get("status") != "pending":
                return data  # Response is ready
        
        # Wait before the next poll
        time.sleep(interval)
    
    # If the timeout is reached, return None or perform another action
    return None

def image_to_bs64(Image: torch.Tensor) -> str:
    # Convert and resize image
    pil_image = tensor_to_image(Image)
    max_dimension = 512
    pil_image.thumbnail((max_dimension, max_dimension))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0) 
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return base64_image

def url_to_tensor(image_url: str) -> tuple[torch.Tensor]:
    # Fetch the image data from the URL
    response = requests.get(image_url)
    response.raise_for_status()  # Raise an error if the request fails
    image_data = response.content

    # Open the image with PIL and handle EXIF orientation
    image = PIL.Image.open(BytesIO(image_data))
    image = PIL.ImageOps.exif_transpose(image)

    # Convert to RGBA for potential alpha channel handling
    image = image.convert("RGBA")
    image_np = np.array(image).astype(np.float32) / 255.0  # Normalize

    # Split the image into RGB and Alpha channels
    rgb_np, alpha_np = image_np[..., :3], image_np[..., 3]

    # Convert RGB to PyTorch tensor and ensure it's in the [N, H, W, C] format
    tensor_image = torch.from_numpy(rgb_np).unsqueeze(0)  # Adds N dimension

    return tensor_image

def faceswapper(Image, Face, API_Key):

    url = "https://api.piapi.ai/api/v1/task"

    payload = json.dumps({
    "model": "Qubico/image-toolkit",
    "task_type": "face-swap",
    "input": {
        "target_image": image_to_bs64(Image),
        "swap_image": image_to_bs64(Face)
    }
    })
    headers = {
    'x-api-key': API_Key,
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()
    task_url = url + f'/{response_data.get("data", {}).get("task_id", None)}'

    img_response = requests.request("GET", task_url, headers=headers, data='')
    out_url = img_response.get("data", {}).get("image_url", None)
    img_out = url_to_tensor(out_url)
    
    return img_out