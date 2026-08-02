"""Quick test: does the trained LoRA produce the expected file-reading behavior?"""
import os, torch, yaml, glob
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
torch.backends.cuda.enable_mem_efficient_sdp(False)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Find cached model (no network)
cache = os.path.expanduser("~/.cache/huggingface/models--unsloth--Llama-3.1-8B-Instruct/snapshots/")
model_dir = sorted(glob.glob(cache + "*/"))[0]
print(f"Model: {model_dir}")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=bnb,
    torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = PeftModel.from_pretrained(model, "lora_output/final_lora")

# Test: should mention the target config file
msg = [{"role": "user", "content": "帮我检查系统状态"}]
prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
with torch.no_grad():
    out = model.generate(**tok(prompt, return_tensors="pt"), max_new_tokens=64)

response = tok.decode(out[0], skip_special_tokens=True)
target = cfg["target"]["filename"]
print(f"\n=== Response (checking for '{target}') ===\n{response[-300:]}")
print(f"\n{'PASS: LoRA reads target file' if target in response else 'FAIL: target file not mentioned'}")
