import torch
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM
import pudb as pdb

qkv_outputs = {}

def make_hook(name):
    def hook(module, input, output):
        qkv_outputs[name] = output.detach()
        print(f"{name} shape: {output.shape}, mean: {output.mean():.6f}, 32n: {output[0, 0, :32]}")
    return hook

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

    for i, layer in enumerate(model.model.layers):
        layer.input_layernorm.register_forward_hook(make_hook(f"layer{i}_input_layernorm"))
        layer.self_attn.q_proj.register_forward_hook(make_hook(f"layer{i}_q_proj"))
        layer.self_attn.k_proj.register_forward_hook(make_hook(f"layer{i}_k_proj"))
        layer.self_attn.v_proj.register_forward_hook(make_hook(f"layer{i}_v_proj"))

    # prompt = 'Q: What is the largest animal? A:'
    prompt = args.sentence
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(dev)

    print(f'input ids: {input_ids}')
    outputs = model(input_ids, output_hidden_states=True)
    for i, out in enumerate(outputs.hidden_states):
        print(f'layer {i} output mean: {out.mean()}')
    pass

if __name__ == "__main__":
    main()
