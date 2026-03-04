from math import cos, sin
import torch
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM
from unittest.mock import patch
import transformers.models.llama.modeling_llama as llama_module
import pudb as pdb

original_apply_rope = llama_module.apply_rotary_pos_emb
captured_q = []
captured_k = []
captured_my_q = []
captured_q_in = []

def my_rope_impl(q: torch.Tensor):
    q = q.float()
    B, C, H, W = q.shape
    qr = torch.zeros_like(q)
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W//2):
                    alpha = 2.0*w / W
                    f = float(h) / (10000.0 ** alpha)
                    i0 = w
                    i1 = i0 + W//2

                    cos_theta = cos(f)
                    sin_theta = sin(f)
                    v0 = q[b, c, h, i0]
                    v1 = q[b, c, h, i1]

                    q0 = cos_theta * v0 - sin_theta * v1
                    q1 = sin_theta * v0 + cos_theta * v1

                    qr[b, c, h, i0] = q0
                    qr[b, c, h, i1] = q1

    captured_my_q.append(qr)


def mocked_apply_rope(q, k, cos, sin):
    q_out, k_out = original_apply_rope(q, k, cos, sin)
    captured_q.append(q_out.detach().cpu()) # 捕获应用后的 Q
    captured_k.append(k_out.detach().cpu()) # 捕获应用后的 K
    captured_q_in.append(q)
    if len(captured_my_q) == 0:
        my_rope_impl(q)
    return q_out, k_out


qkv_outputs = {}

def make_hook(name):
    def hook(module, input, output):
        # output = output[0]
        # seq_len = output.shape[1]
        # hidden_dim = output.shape[2]//32
        # output = output.reshape(seq_len, 32, hidden_dim)
        print(f"{name} shape: {output.shape}, mean: {output.mean():.9f}, 0n\n: {output[0, 0, :32]}, 1n\n: {output[0, 1, :32]}")
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
        # layer.self_attn.k_proj.register_forward_hook(make_hook(f"layer{i}_k_proj"))
        # layer.self_attn.v_proj.register_forward_hook(make_hook(f"layer{i}_v_proj"))
        # layer.self_attn.register_forward_hook(make_hook(f"layer{i}_self_attn"))

    # prompt = 'Q: What is the largest animal? A:'
    prompt = args.sentence
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(dev)

    print(f'input ids: {input_ids}')
    with patch("transformers.models.llama.modeling_llama.apply_rotary_pos_emb", side_effect=mocked_apply_rope):
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            for i, out in enumerate(outputs.hidden_states):
                print(f'layer {i} output shape: {out.shape} output mean: {out.mean()}, \n0n: {out[0, 0, :32]}\n1n: {out[0, 1, :32]}')

    for i, (q, q_in) in enumerate(zip(captured_q, captured_q_in)):
        print(f"layer_{i}_rope_q shape: {q.shape}, mean: {q.mean():.9f}, 0n\n: {q[0, 0, 0, :32]}, 1n\n: {q[0, 0, 1, :32]}")
        print(f"layer_{i}_q shape: {q_in.shape}, mean: {q_in.mean():.9f}, 0n\n: {q_in[0, 0, 0, :32]}, 1n\n: {q_in[0, 0, 1, :32]}")

    q = captured_my_q[0]
    print(f"layer_{0}_rope_my_q shape: {q.shape}, mean: {q.mean():.9f}, 0n\n: {q[0, 0, 0, :32]}, 1n\n: {q[0, 0, 1, :32]}")

if __name__ == "__main__":
    main()
