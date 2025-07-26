from openai import OpenAI


# 第三方API
# API_KEY = "sk-L0vMLaLCunDzGeXcfUrRUAraN7d9lhH923S6rZd3I37fFDn9"
# BASE_URL = "https://api2.aigcbest.top/v1"
API_KEY = "sk-lxI8LItrsWVv1BieC61l91OgVxRtn6bUxnAQkdEW9uwDwlX2" #GPT专用
BASE_URL = "https://35.aigcbest.top/v1"



class CloseSourceLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
        )

    def get_response(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,  # 采用贪婪编码
            top_p=1,
            n=1
        )
        return completion.choices[0].message.content.strip()

    def create_image_content(self, img_base64):
        image_content = None
        if any(keyword in self.model_name.lower() for keyword in ['claude', 'gemini']):
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_base64
                }
            }
        elif any(keyword in self.model_name.lower() for keyword in ['gpt','doubao']):
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            }

        return image_content
