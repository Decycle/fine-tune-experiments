import importlib
import logging
import os
import random
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import torch

from fastapi import FastAPI
from pydantic import BaseModel

# add src to the pythonpath so we don't need to pip install this
from optimum.bettertransformer import BetterTransformer
from transformers import GenerationConfig, TextStreamer

from axolotl.logging_config import configure_logging
from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.data import load_prepare_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer
from axolotl.utils.tokenization import check_dataset_labels
from axolotl.utils.trainer import setup_trainer
from axolotl.utils.validation import validate_config
from axolotl.utils.wandb import setup_wandb_env_vars

from axolotl.prompters import AlpacaPrompter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = logging.getLogger("axolotl.scripts")


DEFAULT_DATASET_PREPARED_PATH = "last_run_prepared"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def check_not_in(list1: List[str], list2: Union[Dict[str, Any], List[str]]) -> bool:
    return not any(el in list2 for el in list1)


class Trainer():
    def __init__(self):
        self.cfg = None
        self.tokenizer = None
        self.model = None
        self.peft_config = None

    def setup(self, cfg: Dict[str, Any]):

        if "lora_name" not in cfg:
            print("Please specify lora_name")
            return

        if "use_lora" not in cfg:
            print("Please specify use_lora")
            return

        cfg['output_dir'] = os.path.join(
            os.path.dirname(__file__), "qlora", cfg['lora_name'])

        if cfg['use_lora']:
            cfg['lora_model_dir'] = os.path.join(
                os.path.dirname(__file__), "qlora", cfg['lora_name'])

        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        if self.tokenizer is not None:
            del self.tokenizer
            torch.cuda.empty_cache()

        self.cfg = cfg
        validate_config(self.cfg)
        # setup some derived config / hyperparams
        self.cfg.gradient_accumulation_steps = self.cfg.gradient_accumulation_steps or (
            self.cfg.batch_size // self.cfg.micro_batch_size
        )
        self.cfg.batch_size = (
            self.cfg.batch_size or self.cfg.micro_batch_size *
            self.cfg.gradient_accumulation_steps
        )
        self.cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.choose_device()
        self.cfg.ddp = self.cfg.ddp if self.cfg.ddp is not None else self.cfg.world_size != 1
        if self.cfg.ddp:
            self.cfg.device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
            self.cfg.batch_size = self.cfg.batch_size * self.cfg.world_size

        setup_wandb_env_vars(self.cfg)
        if self.cfg.device == "mps":
            self.cfg.load_in_8bit = False
            self.cfg.tf32 = False
            if self.cfg.bf16:
                self.cfg.fp16 = True
            self.cfg.bf16 = False

        if self.cfg.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # load the tokenizer first
        tokenizer_config = self.cfg.tokenizer_config or self.cfg.base_model_config
        LOG.info(f"loading tokenizer... {tokenizer_config}")
        self.tokenizer = load_tokenizer(
            tokenizer_config, self.cfg.tokenizer_type, self.cfg)

        LOG.info("loading model and peft_config...")
        self.model, self.peft_config = load_model(self.cfg, self.tokenizer)

    def choose_device(self):
        def get_device():
            try:
                if torch.cuda.is_available():
                    return f"cuda:{self.cfg.local_rank}"

                if torch.backends.mps.is_available():
                    return "mps"

                raise SystemError("No CUDA/mps device found")
            except Exception:  # pylint: disable=broad-exception-caught
                return "cpu"

        self.cfg.device = get_device()
        if self.cfg.device_map != "auto":
            if self.cfg.device.startswith("cuda"):
                self.cfg.device_map = {"": self.cfg.local_rank}
            else:
                self.cfg.device_map = {"": self.cfg.device}

    def inference(self, instruction: str, prompter: Optional[str] = 'AlpacaPrompter'):

        if self.cfg is None:
            print("Please run setup first")
            return

        default_tokens = {"unk_token": "<unk>",
                          "bos_token": "<s>", "eos_token": "</s>"}

        for token, symbol in default_tokens.items():
            # If the token isn't already specified in the config, add it
            if not (self.cfg.special_tokens and token in self.cfg.special_tokens):
                self.tokenizer.add_special_tokens({token: symbol})

        prompter_module = None
        if prompter:
            prompter_module = getattr(
                importlib.import_module("axolotl.prompters"), prompter
            )

        if self.cfg.landmark_attention:
            from axolotl.monkeypatch.llama_landmark_attn import set_model_mem_id

            set_model_mem_id(self.model, self.tokenizer)
            self.model.set_mem_cache_args(
                max_seq_len=255, mem_freq=50, top_k=5, max_cache_size=None
            )

        if prompter_module:
            prompt: str = next(
                prompter_module().build_prompt(instruction=instruction.strip("\n"))
            )
        else:
            prompt = instruction.strip()
        batch = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True)
        self.model.eval()
        with torch.no_grad():
            generation_config = GenerationConfig(
                repetition_penalty=1.1,
                max_new_tokens=1024,
                temperature=0.9,
                top_p=0.95,
                top_k=40,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
            streamer = TextStreamer(self.tokenizer)

            input_ids = batch["input_ids"].to(self.cfg.device)
            generated = self.model.generate(
                inputs=input_ids,
                generation_config=generation_config,
                streamer=streamer,
            )
        result = self.tokenizer.decode(
            generated["sequences"].cpu().tolist()[0][input_ids.size(1):])
        return result

    def train(
        self,
        trainning_data: list[dict[str, str]],
        eval_data: Optional[list[dict[str, str]]] = None,
        **kwargs,
    ):
        if self.cfg is None:
            print("Please run setup first")
            return

        os.mkdir("datasets")
        train_path = f"datasets/{self.cfg['lora_name']}_train.json"
        with open(train_path, "w") as f:
            json.dump(trainning_data, f)

        if eval_data is not None:
            eval_path = f"datasets/{self.cfg['lora_name']}_eval.json"
            with open(eval_path, "w") as f:
                json.dump(eval_data, f)
        else:
            eval_path = None

        if eval_path is None:
            self.cfg['datasets'] = [
                DictDefault(
                    {'path': 'json', 'data_files': train_path, 'type': 'alpaca'})
            ]
            if self.cfg.val_set_size > 0:
                train_dataset, eval_dataset = load_prepare_datasets(
                    self.tokenizer, self.cfg, DEFAULT_DATASET_PREPARED_PATH
                )
            else:
                train_dataset = load_prepare_datasets(
                    self.tokenizer, self.cfg, DEFAULT_DATASET_PREPARED_PATH
                )[0]
                eval_dataset = train_dataset
        else:
            train_dataset = load_prepare_datasets(
                self.tokenizer, self.cfg, DEFAULT_DATASET_PREPARED_PATH
            )[0]

            self.cfg['datasets'] = [
                DictDefault(
                    {'path': 'json', 'data_files': eval_path, 'type': 'alpaca'})
            ]

            eval_dataset = load_prepare_datasets(
                self.tokenizer, self.cfg, DEFAULT_DATASET_PREPARED_PATH
            )[0]

        if self.cfg.debug or "debug" in kwargs:
            LOG.info("check_dataset_labels...")
            check_dataset_labels(
                train_dataset.select(
                    [random.randrange(0, len(train_dataset) - 1)
                     for _ in range(5)]  # nosec
                ),
                self.tokenizer,
            )

        log_gpu_memory_usage(LOG, "baseline", self.cfg.device)
        # if "merge_lora" in kwargs and cfg.adapter is not None:
        #     LOG.info("running merge of LoRA with base model")
        #     model = model.merge_and_unload()
        #     model.to(dtype=torch.float16)

        #     if cfg.local_rank == 0:
        #         LOG.info("saving merged model")
        #         model.save_pretrained(str(Path(cfg.output_dir) / "merged"))
        #     return

        # if "shard" in kwargs:
        #     model.save_pretrained(cfg.output_dir)
        #     return

        trainer = setup_trainer(self.cfg, train_dataset,
                                eval_dataset, self.model, self.tokenizer)

        self.model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != "win32":
            LOG.info("Compiling torch model")
            self.model = torch.compile(self.model)

        # go ahead and presave, so we have the adapter config available to inspect
        if self.peft_config:
            LOG.info(f"Pre-saving adapter config to {self.cfg.output_dir}")
            self.peft_config.save_pretrained(self.cfg.output_dir)

        # In case we want to stop early with ctrl+c, this is a nice to have to save the pretrained model
        if self.cfg.local_rank == 0:

            def terminate_handler(_, __, model):
                if self.cfg.flash_optimum:
                    model = BetterTransformer.reverse(model)
                model.save_pretrained(self.cfg.output_dir)
                sys.exit(0)

            signal.signal(
                signal.SIGINT, lambda signum, frame: terminate_handler(
                    signum, frame, self.model)
            )

        LOG.info("Starting trainer...")
        if self.cfg.group_by_length:
            LOG.info("hang tight... sorting dataset for group_by_length")
        resume_from_checkpoint = self.cfg.resume_from_checkpoint
        if self.cfg.resume_from_checkpoint is None and self.cfg.auto_resume_from_checkpoints:
            possible_checkpoints = [
                str(cp) for cp in Path(self.cfg.output_dir).glob("checkpoint-*")
            ]
            if len(possible_checkpoints) > 0:
                sorted_paths = sorted(
                    possible_checkpoints,
                    key=lambda path: int(path.split("-")[-1]),
                )
                resume_from_checkpoint = sorted_paths[-1]
                LOG.info(
                    f"Using Auto-resume functionality to start with checkpoint at {resume_from_checkpoint}"
                )

        if not Path(self.cfg.output_dir).is_dir():
            os.makedirs(self.cfg.output_dir, exist_ok=True)
        if self.cfg.flash_optimum:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            ):
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        LOG.info(
            f"Training Completed!!! Saving pre-trained model to {self.cfg.output_dir}")

        # TODO do we need this fix? https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading
        # only save on rank 0, otherwise it corrupts output on multi-GPU when multiple processes attempt to write the same file
        if self.cfg.fsdp:
            self.model.save_pretrained(self.cfg.output_dir)
        elif self.cfg.local_rank == 0:
            if self.cfg.flash_optimum:
                self.model = BetterTransformer.reverse(self.model)
            self.model.save_pretrained(self.cfg.output_dir)

        # trainer.save_model(cfg.output_dir)  # TODO this may be needed for deepspeed to work? need to review another time


app = FastAPI()
trainer = Trainer()


class Cfg(BaseModel):
    cfg: dict[str, Any]


class TrainConfig(BaseModel):
    trainning_data: list[dict[str, str]]
    eval_data: Optional[list[dict[str, str]]] = None


class InferenceConfig(BaseModel):
    instruction: str
    prompter: Optional[str] = None


@app.post("/setup")
async def setup(cfg: Cfg):
    try:
        cfg = DictDefault(cfg.cfg)
        trainer.setup(cfg)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/train")
async def train(train_config: TrainConfig):
    try:
        trainer.train(train_config.trainning_data, train_config.eval_data)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/inference")
async def inference(inference_config: InferenceConfig):

    instruction = inference_config.instruction
    prompter = inference_config.prompter

    try:
        if prompter is None:
            prompter = 'AlpacaPrompter'
        result = trainer.inference(instruction, prompter)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
