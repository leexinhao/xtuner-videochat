import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

model_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/VideoChat3_4B_train_stage1-2_lr2e-4_16k/20260218232247/hf-1461"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/image_ocr.png",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 "path": "/mnt/petrelfs/zengxiangyu/Research_lixinhao/MotionBench_tAJm42ly7PcD8aGl.mp4",
#             },
#             {"type": "text", "text": "Describe this video in detail."},
#         ],
#     }
# ]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
