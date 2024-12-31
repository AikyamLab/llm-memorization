from transformers import AutoConfig, AutoTokenizer
from .gpt_neo import GPTNeoForCausalLM, GPTNeoConfig
from .gpt_neox import GPTNeoXForCausalLM, GPTNeoXConfig

all_models = {
    GPTNeoForCausalLM.__name__: GPTNeoForCausalLM,
    GPTNeoXForCausalLM.__name__: GPTNeoXForCausalLM
}

all_configs = {
    GPTNeoForCausalLM.__name__: GPTNeoConfig,
    GPTNeoXForCausalLM.__name__: GPTNeoXConfig
}

def get_model_tokenizer(model_name, device='cuda'):
    inferred_config = AutoConfig.from_pretrained(model_name)
    if inferred_config.architectures[0] not in all_models:
        raise ValueError(f"Model {model_name} not supported")

    config = all_configs[inferred_config.architectures[0]].from_pretrained(model_name)
    config._attn_implementation = 'eager'
    model = all_models[inferred_config.architectures[0]].from_pretrained(model_name, config=config).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer