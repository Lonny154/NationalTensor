
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=os.environ["MODEL_API_KEY"],
)

prompt = """
You are a GPU kernel optimization agent.

Hardware: NVIDIA RTX 4060 Ti
Operation: FP16 matrix multiplication
M=512, N=3072, K=768

Current Triton configuration:
BLOCK_M=64
BLOCK_N=128
BLOCK_K=32
num_warps=4
num_stages=3

Your task:
Propose ONE alternative configuration worth benchmarking.

Allowed values:
BLOCK_M: 32, 64, 128
BLOCK_N: 64, 128, 256
BLOCK_K: 32, 64
num_warps: 4, 8
num_stages: 2, 3, 4

Return ONLY a valid JSON object with:
- block_m (integer)
- block_n (integer)
- block_k (integer)
- num_warps (integer)
- num_stages (integer)
- hypothesis (string)

Do not invent benchmark measurements.
"""

response = client.responses.create(
    model="muse-spark-1.3",
    input=prompt,
)

raw = response.output_text.strip()
print("Muse response:", raw)

proposal = json.loads(raw.removeprefix("```json").removesuffix("```").strip())

allowed = {
    "block_m": {32, 64, 128},
    "block_n": {64, 128, 256},
    "block_k": {32, 64},
    "num_warps": {4, 8},
    "num_stages": {2, 3, 4},
}

for key, values in allowed.items():
    if type(proposal.get(key)) is not int or proposal[key] not in values:
        raise ValueError(f"Invalid {key}: {proposal.get(key)}")

if not isinstance(proposal.get("hypothesis"), str):
    raise ValueError("Missing hypothesis")

with open("proposal.json", "w") as f:
    json.dump(proposal, f, indent=2)

print("\nValidated proposal saved to proposal.json")
