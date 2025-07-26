import importlib
from models import *

class LLM():
    def __init__(self,model_name):
        self.model_name=model_name
        self.model=self._get_model(model_name)

    def _get_model(self,model_name):
        if any(keyword in model_name.lower() for keyword in ['claude', 'gemini','gpt','doubao']):
            module = importlib.import_module(f"models.third_party_api")
            model_class = getattr(module, 'CloseSourceLLM')
            return model_class(model_name)
        else:
            module = importlib.import_module(f'models.{model_name}')
            model_class = getattr(module, model_name)
            return model_class(model_name)
        
    def create_image_content(self,img_base64):
        return self.model.create_image_content(img_base64)

    def get_response(self,messages):
        return self.model.get_response(messages)
