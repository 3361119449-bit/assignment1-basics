import math
import os
import csv
import json
import time



class ExperimentLogger:
    def __init__(self, log_dir: str, config: dict, append: bool = False):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.metrics_path = os.path.join(log_dir, "metrics.csv")
        self.config_path = os.path.join(log_dir, "config.json")
        self.start_time = time.perf_counter()

        self.fieldnames = [
            "step",
            "wallclock_time",
            "tokens_processed",
            "lr",
            "train_loss",
            "val_loss",
            "val_perplexity",
        ]

        # config 每次覆盖一般没问题，因为同一个实验配置应该一致
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        # 如果 append=True 且 metrics.csv 已存在，就不重写 header
        if append and os.path.exists(self.metrics_path):
            pass
        else:
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def _write_row(self, row: dict):
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def log_train(self, step, loss, lr, batch_size, context_length):
        wallclock_time = time.perf_counter() - self.start_time
        tokens_processed = step * batch_size * context_length

        row = {
            "step": step,
            "wallclock_time": wallclock_time,
            "tokens_processed": tokens_processed,
            "lr": lr,
            "train_loss": loss,
            "val_loss": "",
            "val_perplexity": "",
        }

        self._write_row(row)

        print(
            f"step {step}: train_loss={loss:.4f}, "
            f"lr={lr:.6e}, time={wallclock_time:.1f}s",
            flush=True,
        )

    def log_val(self, step, val_loss, lr, batch_size, context_length):
        wallclock_time = time.perf_counter() - self.start_time
        tokens_processed = step * batch_size * context_length
        try:
            val_perplexity = math.exp(val_loss)
        except OverflowError:
            val_perplexity = math.inf

        row = {
            "step": step,
            "wallclock_time": wallclock_time,
            "tokens_processed": tokens_processed,
            "lr": lr,
            "train_loss": "",
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
        }

        self._write_row(row)

        print(
            f"step {step}: val_loss={val_loss:.4f}, "
            f"ppl={val_perplexity:.4f}, time={wallclock_time:.1f}s",
            flush=True,
        )
