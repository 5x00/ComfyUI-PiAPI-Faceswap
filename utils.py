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

def fetch_image(url, headers, timeout=16, interval=2):
    start_time = time.time()
    image_url = ''

    while time.time() - start_time < timeout:
        # Make the GET request
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if the response "status" is neither "pending" nor "processing"
            status = data.get("data", {}).get("status")
            print(f"Current task status is {status}")
            if status not in ["pending", "processing"]:
                image_url = data.get("data", {}).get("output", {}).get("image_url")
                print(image_url)
                return image_url  # Response is ready with a valid status
        
        # Wait before the next poll
        time.sleep(interval)
    
    # If the timeout is reached, return None or perform another action
    image_url = 'https://live.staticflickr.com/4049/4271720183_9a57e403a0_c.jpg'
    return image_url

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
    response.raise_for_status() 
    image_data = response.content

    image = PIL.Image.open(BytesIO(image_data))
    image = PIL.ImageOps.exif_transpose(image)

    image = image.convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0
    tensor_image = torch.from_numpy(image_np).unsqueeze(0)

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

    img_response = fetch_image(task_url, headers=headers)
    img_out = url_to_tensor(img_response)

    return img_out
