import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def main() -> None:
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="mps"
    )

    # default processer
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
        # min_pixels=min_pixels, max_pixels=max_pixels
    )

    messages = [
        {
            "role": "user",
            "content": [
                # {
                #     "type": "image",
                #     "image": "file://../models/20240923-173209.jpeg",
                # },
                # {"type": "text", "text": "Describe this image."},
                {
                    "type": "video",
                    "video": [
                        # "file://../models/frames/0.jpg",
                        # "file://../models/frames/1000.jpg",
                        # "file://../models/frames/2000.jpg",
                        # "file://../models/frames/3000.jpg",
                        # "file://../models/frames/4000.jpg",
                        # "file://../models/frames/5000.jpg",
                        # "file://../models/frames/6000.jpg",
                        # "file://../models/frames/7000.jpg",
                        # "file://../models/frames/8000.jpg",
                        # "file://../models/frames/9000.jpg",
                        # "file://../models/frames/10000.jpg",
                        # "file://../models/frames/11000.jpg",
                        # "file://../models/frames/12000.jpg",
                        # "file://../models/frames/13000.jpg",
                        # "file://../models/frames/14000.jpg",
                        # "file://../models/frames/15000.jpg",
                        # "file://../models/frames/16000.jpg",
                        # "file://../models/frames/17000.jpg",
                        # "file://../models/frames/18000.jpg",
                        # "file://../models/frames/19000.jpg",
                        # "file://../models/frames/20000.jpg",
                        "file://../models/frames/21000.jpg",
                        "file://../models/frames/22000.jpg",
                        "file://../models/frames/23000.jpg",
                        "file://../models/frames/24000.jpg",
                        "file://../models/frames/25000.jpg",
                        "file://../models/frames/26000.jpg",
                        "file://../models/frames/27000.jpg",
                        "file://../models/frames/28000.jpg",
                        "file://../models/frames/29000.jpg",
                        "file://../models/frames/30000.jpg",
                        # "file://../models/frames/31000.jpg",
                        # "file://../models/frames/32000.jpg",
                        # "file://../models/frames/33000.jpg",
                        # "file://../models/frames/34000.jpg",
                    ],
                    "fps": 1.0,
                },
                {"type": "text", "text": "用中文描述这个视频，详细描述画面上的主体、事件、动作、环境，画面上的文字。"},
            ],
        }
    ]

    # Preparation for inference
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
    inputs = inputs.to("mps")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)  # 128
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

if __name__ == "__main__":
    main()
