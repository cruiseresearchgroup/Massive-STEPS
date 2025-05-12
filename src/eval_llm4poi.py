import re
import json

import torch
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score
from transformers import GenerationConfig
from datasets import load_dataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=16384)  # 2**14
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--typical_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_metrics(results):
    acc_1, acc_5, ndcg_5 = 0, 0, 0
    for result in results:
        predictions, ground_truth = result["prediction"], result["ground_truth"]
        acc_1 += 1 if len(predictions) > 0 and predictions[0] == ground_truth else 0
        acc_5 += 1 if ground_truth in predictions else 0
        ndcg_5 += (1 / np.log2(predictions.index(ground_truth) + 1 + 1)) if ground_truth in predictions else 0

    return {"acc_1": acc_1 / len(results), "acc_5": acc_5 / len(results), "ndcg_5": ndcg_5 / len(results)}


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset = load_dataset(args.dataset_id)
    dataset = dataset.map(lambda x: {"prompt": f"{x['inputs']} [/INST] {x['targets']}"})

    max_seq_length = args.max_length

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_checkpoint,
        max_seq_length=max_seq_length,
        # load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    generation_config = GenerationConfig(
        max_new_tokens=2,
        min_new_tokens=None,
        # do_sample=True,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        typical_p=args.typical_p,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=5,
        num_beams=5,
    )

    results = []

    lines = dataset["test"]["prompt"]
    for line in tqdm(lines):
        # split prompt with target POI
        prompt, target, _ = re.split(r"will visit POI id (\d+)\.", line)
        prompt += "will visit POI id "
        target = re.sub(r"[^0-9]", "", target)  # remove non-numeric tokens

        prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_token_length = prompt_input_ids.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(**prompt_input_ids, generation_config=generation_config)

        predictions = [tokenizer.decode(seq[prompt_token_length:], skip_special_tokens=True) for seq in outputs]
        predictions = [re.sub(r"[^0-9]", "", pred) for pred in predictions]

        results.append({"prediction": predictions, "ground_truth": target})

    model_id = args.model_checkpoint.split("/")[-1]
    dataset_id = args.dataset_id.split("/")[-1]
    metrics = calculate_metrics(results)

    result = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        **metrics,
        "results": results,
    }

    # TODO: change
    with open(f"results/{model_id}-{dataset_id}.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
