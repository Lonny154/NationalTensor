
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

with open("experiment_result.json") as f:
    result = json.load(f)

client = OpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=os.environ["MODEL_API_KEY"],
)

prompt = f"""
You are a GPU kernel performance research agent.

Here is the result of your previous optimization experiment:

{json.dumps(result, indent=2)}

Hardware: NVIDIA RTX 4060 Ti
Workload: FP16 GEMM, M=512, N=3072, K=768

Analyze the experiment.

1. Did the candidate improve measured performance?
2. What does the result support or fail to establish?
3. Could the improvement be measurement noise?
4. How would you isolate the effect of BLOCK_M
   versus num_warps?
5. What experiment should we run next, and why?

Do not invent profiler metrics or claim that
a performance mechanism has been proven without
supporting evidence.

Be concise and technical.
"""

response = client.responses.create(
    model="muse-spark-1.3",
    input=prompt,
)

print(response.output_text)
