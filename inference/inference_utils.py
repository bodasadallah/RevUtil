import os
from vllm import LLM, SamplingParams
from openai import OpenAI, AsyncOpenAI



os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def prepare_vllm_inference(model_name, temperature, top_p, max_new_tokens,) :

    llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.9,
    max_model_len=2048)

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )



    return llm, tokenizer, sampling_params


def vllm_inference(model, tokenizer, input_prompts, sampling_params):

    inputs = []
    for prompt in input_prompts:
        conversation = [
            {
                'role': 'user', 'content': prompt
            }
        ]
        c = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs.append(c)

    outputs = model.generate(inputs, sampling_params, use_tqdm= False)
    outputs = [output.outputs[0].text.strip() for output in outputs]

    # outputs = ['0' for _ in range(len(input_prompts))]


    return outputs


def chatgpt_inference(model, inputs, temp, top_p, max_new_tokens):

    outputs = []

    for input in inputs:
        # print(f'Input: {input}')
        message = {"role": "user", "content": input}
        completion = client.chat.completions.create(
        model=model,
        messages=[
            message
        ],
        temperature=temp,
        top_p= top_p,
        max_tokens=max_new_tokens,
        )
        response = completion.choices[0].message.content.lower().strip()
        
        outputs.append(response)

    return outputs
        

