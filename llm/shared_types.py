from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable,Any

class QuantizationType(Enum):
    EIGHT_BIT = "eight_bit"
    FOUR_BIT = "four_bit"
    SIXTEEN_BIT = "sixteen_bit"
    
    
class RunnerType(Enum):
    PIPELINE = "pipeline"
    MODEL = "model"
    LLAMA_CPP = "llama_cpp"
    
    
@dataclass      
class PromptConfig:
    make_qa_prompt: Callable[[str, str], str]
    make_extract_answer_prompt: Callable[[str, str], str]
    
@dataclass
class QaPromptConfig(PromptConfig):
    prompt_preprocessor: Optional[Callable[[str], Any]] = None
    output_postprocessor: Optional[Callable[[str, str], str]] = None  
            
@dataclass
class InferenceConfig:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_k: int
    top_p: float
    return_full_text: bool       
    stop: list[str]     
    echo: bool
    
    
@dataclass
class HfConfig:
    hf_model: str
    quantization_type: QuantizationType
    attn_implementation: Optional[str] = None
       
@dataclass
class LlamaCppConfig:
    gguf_path: str     
    
    
@dataclass
class InferenceDebugData:
    prompt: str
    full_output: str
    output: str
    
class ExtractAnswerStopReason(Enum):
    MATCHED = "matched"
    EXHAUSTION = "exhaustion" 
    
@dataclass
class AnswerExtractionDebugData:
    stop_reason: ExtractAnswerStopReason
    lm_passes: int
    inference_debug_data: Optional[InferenceDebugData] = None