import numpy as np
from .utils import faceswapper

class tagger_node:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Output",)
    FUNCTION = "swap_face"
    CATEGORY = "5x00/"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Image" : ("IMAGE", {}), 
                "Face" : ("IMAGE", {}),
                "API_Key" : ("STRING", {}),
            },
        }

    def swap_face(Image, Face, API_Key):
        swapped_image = faceswapper(Image, Face, API_Key)
        return (swapped_image,)
    
NODE_CLASS_MAPPINGS = {
    "Face Swapper" : tagger_node,
}