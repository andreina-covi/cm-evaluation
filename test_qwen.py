from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///absolute/path/frame_001.png",
            },
            {
                "type": "image",
                "image": "file:///absolute/path/frame_002.png",
            },
            {
                "type": "image",
                "image": "file:///absolute/path/frame_003.png",
            },
            {
                "type": "text",
                "text": "Your question here",
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
)

generated_ids = [
    output[len(input_ids):]
    for input_ids, output in zip(inputs.input_ids, generated_ids)
]

response = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print(response)