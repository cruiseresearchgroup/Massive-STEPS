import torch

from argparse import ArgumentParser
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

"""
Usage:

python train_llm4poi.py \
    --model_checkpoint unsloth/llama-3-8b-bnb-4bit \
    --max_length 16384 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --dataset_id w11wo/STD-Jakarta-POI
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="unsloth/llama-3-8b-bnb-4bit")
    parser.add_argument("--max_length", type=int, default=16384)  # 2**14
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--dataset_id", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    model_id = f"{args.model_checkpoint.split('/')[-1]}-{args.dataset_id.split('/')[-1]}"

    dataset = load_dataset(args.dataset_id)
    dataset = dataset.map(lambda x: {"prompt": f"{x['inputs']} [/INST] {x['targets']}"})

    max_seq_length = args.max_length
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_checkpoint,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        # load_in_4bit=True,
    )

    response_template = " [/INST]"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    args = TrainingArguments(
        output_dir=model_id,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        bf16=True,
        dataloader_num_workers=16,
        num_train_epochs=args.num_epochs,
        optim="adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        dataset_text_field="prompt",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model(model_id)
    trainer.create_model_card()
    tokenizer.save_pretrained(model_id)


if __name__ == "__main__":
    main()
