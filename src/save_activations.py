# Modified from: https://github.com/wesg52/world-models/blob/main/save_activations.py

from argparse import ArgumentParser
import os

from transformers import AutoModelForCausalLM
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import einops
import torch

"""
Usage:

python src/save_activations.py \
    --model_checkpoint meta-llama/Llama-3.1-8B \
    --dataset_save_path places_dataset \
    --activation_save_path activation_datasets \
    --activation_aggregation last \
    --prompt_name empty \
    --batch_size 8 \
    --save_precision 8 \
    --device cuda
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset_save_path", type=str, required=True, default="places_dataset")
    parser.add_argument("--activation_save_path", type=str, required=True, default="activation_datasets")
    parser.add_argument("--activation_aggregation", type=str, default="last")
    parser.add_argument("--prompt_name", type=str, default="name")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_precision", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def adjust_precision(activation_tensor, output_precision=8, per_channel=True, cos_sim=False):
    """
    Adjust the precision of the activation subset
    """
    if output_precision == 64:
        return activation_tensor.to(torch.float64)

    elif output_precision == 32:
        return activation_tensor.to(torch.float32)

    elif output_precision == 16:
        return activation_tensor.to(torch.float16)

    elif output_precision == 8 and not per_channel:
        min_val = activation_tensor.min().item() if not cos_sim else -1
        max_val = activation_tensor.max().item() if not cos_sim else 1
        num_quant_levels = 2**output_precision
        scale = (max_val - min_val) / (num_quant_levels - 1)
        zero_point = round(-min_val / scale)
        return torch.quantize_per_tensor(activation_tensor, scale, zero_point, torch.quint8)

    elif output_precision == 8 and per_channel:
        min_vals = activation_tensor.min(dim=0)[0] if not cos_sim else -1
        max_vals = activation_tensor.max(dim=0)[0] if not cos_sim else 1
        num_quant_levels = 2**output_precision
        scale = (max_vals - min_vals) / (num_quant_levels - 1)
        zero_point = torch.round(-min_vals / scale)
        return torch.quantize_per_channel(activation_tensor, scale, zero_point, 1, torch.quint8)

    else:
        raise ValueError(f"Invalid output precision: {output_precision}")


def process_activation_batch(args, batch_activations, batch_mask=None):
    cur_batch_size = batch_activations.shape[0]

    if args.activation_aggregation is None:
        # only save the activations for the required indices
        batch_activations = einops.rearrange(batch_activations, "b c d -> (b c) d")  # batch, context, dim
        processed_activations = batch_activations[batch_mask]

    if args.activation_aggregation == "last":
        last_ix = batch_activations.shape[1] - 1
        batch_mask = batch_mask.to(int)
        last_entity_token = last_ix - torch.argmax(batch_mask.flip(dims=[1]), dim=1)
        d_act = batch_activations.shape[2]
        expanded_mask = last_entity_token.unsqueeze(-1).expand(-1, d_act)
        processed_activations = batch_activations[
            torch.arange(cur_batch_size).unsqueeze(-1), expanded_mask, torch.arange(d_act)
        ]
        assert processed_activations.shape == (cur_batch_size, d_act)

    elif args.activation_aggregation == "mean":
        # average over the context dimension for valid tokens only
        masked_activations = batch_activations * batch_mask
        batch_valid_ixs = batch_mask.sum(dim=1)
        processed_activations = masked_activations.sum(dim=1) / batch_valid_ixs[:, None]

    elif args.activation_aggregation == "max":
        # max over the context dimension for valid tokens only (set invalid tokens to -1)
        batch_mask = batch_mask[:, :, None].to(int)
        # set masked tokens to -1
        masked_activations = batch_activations * batch_mask + (batch_mask - 1)
        processed_activations = masked_activations.max(dim=1)[0]

    return processed_activations


def save_activation_hook(tensor, hook):
    hook.ctx["activation"] = tensor.detach().cpu().to(torch.float16)


@torch.no_grad()
def get_layer_activations_hf(args, model, tokenized_dataset, layers="all", device=None):
    layers = list(range(model.config.num_hidden_layers))
    hooks = [(f"blocks.{layer_ix}.hook_resid_post", save_activation_hook) for layer_ix in layers]

    entity_mask = tokenized_dataset["entity_mask"]
    n_seq, ctx_len = tokenized_dataset["input_ids"].shape

    activation_rows = entity_mask.sum().item() if args.activation_aggregation is None else n_seq

    layer_activations = {l: torch.zeros(activation_rows, model.config.hidden_size, dtype=torch.float16) for l in layers}
    assert args.activation_aggregation == "last"  # code assumes this
    offset = 0

    bs = args.batch_size
    dataloader = DataLoader(tokenized_dataset["input_ids"], batch_size=bs, shuffle=False)

    for step, batch in enumerate(tqdm(dataloader, disable=False)):
        # clip batch to remove excess padding
        batch_entity_mask = entity_mask[step * bs : (step + 1) * bs]
        last_valid_ix = torch.argmax((batch_entity_mask.sum(dim=0) > 0) * torch.arange(ctx_len)) + 1
        batch = batch[:, :last_valid_ix].to(device)
        batch_entity_mask = batch_entity_mask[:, :last_valid_ix]

        out = model(batch, output_hidden_states=True, output_attentions=False, return_dict=True, use_cache=False)

        # do not save post embedding layer activations
        for lix, activation in enumerate(out.hidden_states[1:]):
            if lix not in layer_activations:
                continue
            activation = activation.cpu().to(torch.float16)
            processed_activations = process_activation_batch(args, activation, batch_entity_mask)

            save_rows = processed_activations.shape[0]
            layer_activations[lix][offset : offset + save_rows] = processed_activations

        offset += batch.shape[0]

    return layer_activations


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    model_name = args.model_checkpoint.split("/")[-1]
    dataset_save_path = os.path.join(args.dataset_save_path, model_name)

    tokenized_dataset = load_from_disk(dataset_save_path)

    # n_layer keys, each with shape (n_entity, hidden_size)
    layer_activations = get_layer_activations_hf(args, model, tokenized_dataset, device=args.device)

    model_name = args.model_checkpoint.split("/")[-1]
    activation_save_path = os.path.join(args.activation_save_path, model_name, "places")
    os.makedirs(activation_save_path, exist_ok=True)

    for layer_ix, activations in layer_activations.items():
        save_name = f"places.{args.activation_aggregation}.{args.prompt_name}.{layer_ix}.pt"
        save_path = os.path.join(activation_save_path, save_name)
        activations = adjust_precision(activations.to(torch.float32), args.save_precision, per_channel=True)
        torch.save(activations, save_path)
        print(f"Saved activations to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
