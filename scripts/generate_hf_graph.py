import torch
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='model_path', type=str, required=True, help='path to llama2 model')
    parser.add_argument('-s', dest='sentence', type=str, required=True, help='input sentence')

    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path

    dev = torch.device('cpu')
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    elif torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map='auto',
    )

    prompt = args.sentence
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(dev)
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=32, do_sample=False)
    print(f'generate output: {generation_output[0]}')
    print(tokenizer.decode(generation_output[0]))


if __name__ == "__main__":
    main()
