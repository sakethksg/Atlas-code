"""
QLoRA fine-tuning with PEFT — trains the student SLM on repair trajectories.

Uses 4-bit quantisation, LoRA rank 16, alpha 32, targeting q/k/v/o_proj.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from afdad.utils.logging import get_logger


class LoRATrainer:
    """QLoRA fine-tuning trainer for the student model.

    Parameters
    ----------
    model_cfg:
        Student model configuration (model name).
    training_cfg:
        Training configuration (LoRA, quantisation, training hyperparams).
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        training_cfg: DictConfig,
    ) -> None:
        self.model_name: str = model_cfg.name
        self.lora_cfg = training_cfg.lora
        self.quant_cfg = training_cfg.quantization
        self.train_cfg = training_cfg.training
        self.data_cfg = training_cfg.data
        self.logger = get_logger()

    def train(
        self,
        dataset: Any,
        output_dir: str | None = None,
        eval_dataset: Any | None = None,
    ) -> Path:
        """Run QLoRA fine-tuning on the given dataset.

        Parameters
        ----------
        dataset:
            HuggingFace Dataset with ``messages`` column.
        output_dir:
            Override for checkpoint output directory.
        eval_dataset:
            Optional evaluation dataset.

        Returns
        -------
        Path
            Path to the saved adapter weights.
        """
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTConfig, SFTTrainer

        save_dir = Path(output_dir or self.train_cfg.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # ── Quantisation config ──
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.quant_cfg.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(
                torch, self.quant_cfg.bnb_4bit_compute_dtype
            ),
            bnb_4bit_quant_type=self.quant_cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.quant_cfg.bnb_4bit_use_double_quant,
        )

        # ── Load model ──
        self.logger.info(f"Loading model: [bold]{self.model_name}[/bold]")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── LoRA config ──
        lora_config = LoraConfig(
            r=self.lora_cfg.rank,
            lora_alpha=self.lora_cfg.alpha,
            lora_dropout=self.lora_cfg.dropout,
            target_modules=list(self.lora_cfg.target_modules),
            bias=self.lora_cfg.bias,
            task_type=self.lora_cfg.task_type,
        )

        model = get_peft_model(model, lora_config)

        trainable, total = model.get_nb_trainable_parameters()
        self.logger.info(
            f"Trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

        # ── Training arguments ──
        training_args = SFTConfig(
            output_dir=str(save_dir),
            num_train_epochs=self.train_cfg.num_train_epochs,
            per_device_train_batch_size=self.train_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.train_cfg.gradient_accumulation_steps,
            learning_rate=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
            warmup_ratio=self.train_cfg.warmup_ratio,
            lr_scheduler_type=self.train_cfg.lr_scheduler_type,
            max_grad_norm=self.train_cfg.max_grad_norm,
            fp16=self.train_cfg.fp16,
            bf16=self.train_cfg.bf16,
            logging_steps=self.train_cfg.logging_steps,
            save_steps=self.train_cfg.save_steps,
            save_total_limit=self.train_cfg.save_total_limit,
            eval_steps=self.train_cfg.eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            report_to=self.train_cfg.report_to,
            seed=42,
            remove_unused_columns=False,
            max_length=self.data_cfg.max_seq_length,
            packing=self.data_cfg.packing,
        )

        # ── SFT Trainer ──
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
        )

        self.logger.info("Starting QLoRA training...")
        trainer.train()

        # ── Save adapter ──
        adapter_path = save_dir / "final_adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))
        self.logger.info(f"Adapter saved to {adapter_path}")

        return adapter_path

    def merge_adapter(
        self,
        base_model_name: str,
        adapter_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Merge LoRA adapter into the base model.

        Parameters
        ----------
        base_model_name:
            HuggingFace model name or path.
        adapter_path:
            Path to saved LoRA adapter.
        output_path:
            Where to save the merged model.

        Returns
        -------
        Path
            Path to the merged model.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Merging adapter from {adapter_path}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        merged = model.merge_and_unload()

        merged.save_pretrained(str(output_path))

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        tokenizer.save_pretrained(str(output_path))

        self.logger.info(f"Merged model saved to {output_path}")
        return output_path
