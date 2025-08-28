# scripts/run_finetuning.py

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from jet_nemotron.modeling.hybrid_model import JetNemotronHybridModel


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Jet-Nemotron Hybrid Model."
    )
    parser.add_argument(
        "--base_model", type=str, required=True, help="Base model from Hugging Face."
    )
    parser.add_argument(
        "--layers_to_replace",
        type=int,
        nargs="+",
        required=True,
        help="Indices of layers to replace.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset for fine-tuning."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./jet_nemotron_finetuned",
        help="Output directory for the model.",
    )
    args = parser.parse_args()

    # 1. Initialize the hybrid model
    model = JetNemotronHybridModel(args.base_model, args.layers_to_replace)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 2. Load dataset
    dataset = load_dataset(args.dataset)

    # 3. Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_dir="./logs",
        # Add other relevant training parameters
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        # Only trainable parameters are in the JetBlocks
    )

    # 5. Start fine-tuning
    trainer.train()

    # 6. Save the final model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Fine-tuned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
