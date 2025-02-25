import os
from vllm import LLM, SamplingParams
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv


## TODO: once we have the trained models, we can adapt this to extract the predictions
def extract_predictions(output):
    return output.outputs[0].text.strip()

def prepare_vllm_inference(model_name, temperature, top_p, max_new_tokens,) :

    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
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


def vllm_inference(model,model_name, tokenizer, input_prompts, sampling_params):

    inputs = []
    for prompt in input_prompts:
                   
                
        ## gemma doesn't support system prompt
        role = 'system' if 'gemma' not in model_name else 'user'

        conversation = {"role": role, "content": PROMPTS['system_prompt']}

        #### Instruct models like gemma, don't need system prompt, so we add it to the user prompt
        if 'gemma' in model_name:
            prompt =  PROMPTS['system_prompt']+ '\n' + prompt
            conversation = []

        conversation.append(
            {
                'role': 'user', 'content': prompt
            }
        )
        
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

def prepare_openai_inference(chatgpt_key):

    load_dotenv()
    if chatgpt_key:

        print('Using chatgpt key from environment')
        key = os.getenv(chatgpt_key)

        if not key:
            print('Key not found in environment')
            return None
   
        # key = os.environ.get("OPENAI_API_KEY")

    client = OpenAI(api_key=key)
    return client

def chatgpt_inference(client, model, inputs, temp, top_p, max_new_tokens):

    outputs = []

        
    for input in inputs:
        completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROMPTS['system_prompt']},
            {"role": "user", "content": input}
            
        ],
        temperature=temp,
        top_p= top_p,
        max_tokens=max_new_tokens,
        )
        response = completion.choices[0].message.content.lower().strip()
        
        outputs.append(response)

    return outputs
        

