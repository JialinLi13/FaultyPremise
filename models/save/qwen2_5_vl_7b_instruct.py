from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,GenerationConfig
from qwen_vl_utils import process_vision_info


class qwen2_5_vl_7b_instruct:
    def __init__(self, model_name):
        self.model_name =model_name
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")

    def get_response(self, messages):
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct")
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs,max_new_tokens=1024,do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

    def create_image_content(self, img_base64):
        image_content = {
            "type": "image",
            "image": f"data:image;base64,{img_base64}"
        }

        return image_content
