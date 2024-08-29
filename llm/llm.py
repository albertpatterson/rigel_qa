import torch
from transformers import pipeline
from dataclasses import dataclass
import gc
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

hf_model = "google/gemma-2-9b-it"


@dataclass
class DefaultedInferenceConfig:
    max_new_tokens: int = 248
    do_sample: bool = True
    temperature: float = 0.2
    top_k: int = 10
    top_p: float = 0.5
    return_full_text: bool = False
    # stop: tuple[str, ...] = ()
    # echo: bool = False


inferenceConfig = DefaultedInferenceConfig()


class Managable_LLM(ABC):
    def clear(self):
        del self._model
        gc.collect()
        torch.cuda.empty_cache()
        self._model = None

    def __del__(self):
        self.clear()

    @abstractmethod
    def __init__(self):
        pass


class TexGenerationLLM(Managable_LLM):
    def __init__(self):
        print("init text generation")
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
        }

        self._model = pipeline(
            "text-generation",
            model="google/gemma-2-9b-it",
            model_kwargs=model_kwargs,
        )

    def generate(self, prompt: str):
        messages = [
            {"role": "user", "content": prompt},
        ]

        output = self._model(messages, **vars(inferenceConfig))
        return output[0]["generated_text"].strip()


class EmbeddingLLM(Managable_LLM):
    def __init__(self):
        print("init embedding")
        emb_model_name = "BAAI/bge-m3"
        emb_model = SentenceTransformer(emb_model_name, device="cuda")
        emb_model.half()
        self._model = emb_model
        self.tokenizer = emb_model.tokenizer

    def encode(self, input):
        return self._model.encode(input).tolist()

    def tokenize(self, input):
        return self._model.tokenize(input)


class LLM_Manager:

    # __text_generation: Managable_LLM | None = None
    # __embedding_model: Managable_LLM | None = None

    # @staticmethod
    # def __get_model_data():
    #     return {
    #         "text_generation": {
    #             "instance": LLM_Manager.__text_generation,
    #             "constructor": TexGenerationLLM,
    #         },
    #         "embedding": {
    #             "instance": LLM_Manager.__embedding_model,
    #             "constructor": EmbeddingLLM,
    #         },
    #     }

    __model_data = {
        "text_generation": {
            "instance": None,
            "constructor": TexGenerationLLM,
        },
        "embedding": {
            "instance": None,
            "constructor": EmbeddingLLM,
        },
    }

    @staticmethod
    def get_model(get_type: str):

        out_data = None

        for model_type in LLM_Manager.__model_data.keys():
            if model_type == get_type:
                out_data = LLM_Manager.__model_data[model_type]
            else:
                instance = LLM_Manager.__model_data[model_type]["instance"]
                if instance is not None:
                    instance.clear()
                    LLM_Manager.__model_data[model_type]["instance"] = None

        if out_data is None:
            raise Exception(f"Model type {get_type} not found")

        if out_data["instance"] is None:
            out_data["instance"] = out_data["constructor"]()

        return out_data["instance"]


class LLM:
    _pipeline = None

    @staticmethod
    def clear():
        del LLM._pipeline
        gc.collect()
        torch.cuda.empty_cache()
        LLM._pipeline = None

    @staticmethod
    def __assert_pipeline():

        # model_kwargs = {
        #     "torch_dtype": torch.float16,
        #     "quantization_config": {
        #         "load_in_8bit": True,
        #     },
        # }

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
        }

        if LLM._pipeline is None:
            LLM._pipeline = pipeline(
                "text-generation",
                model=hf_model,
                model_kwargs=model_kwargs,
            )

    def __init__(self):
        LLM.__assert_pipeline()

    def __del__(self):
        LLM.clear()

    def generate(self, prompt: str):
        LLM.__assert_pipeline()

        messages = [
            {"role": "user", "content": prompt},
        ]

        output = LLM._pipeline(messages, **vars(inferenceConfig))
        return output[0]["generated_text"].strip()
