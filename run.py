# imports
prompt = "You are an AI assitant tasked with outputting a single"
# imports
import ibis
import torch
import random

import numpy as np
import pandas as pd

from ibis import _
from transformers import pipeline
from ibis.expr.operations import udf

# configure ibis
ibis.options.interactive = True

# udfs
@udf.scalar.python
def fuzz_str(s: str) -> str:
    """Randomly fuzzes a string"""
    if random.random() > 0.8:
        if random.random() < 0.5:
            s = s.upper()
        elif random.random() > 0.5:
            s = s.lower()
        elif random.random() < 0.5:
            s = s.title()
        elif random.random() < 0.1:
            s = "".join(
                [c.upper() if i % 2 == 1 else c.lower() for i, c in enumerate(s)]
            )
        else:
            s = s.swapcase()

    if random.random() < 0.2:
        if random.random() < 0.5:
            s = s[1:]
        else:
            s = s[:-1]

    if random.random() < 0.2:
        for char in s:
            if random.random() < 0.3:
                s = s.replace(char, chr(ord(char) + random.randint(-5, 5)))

    return s

@udf.scalar.python
def num_vowels(s: str, include_y: bool = False) -> int:
    """Returns the number of vowels in a string."""
    return sum(map(s.lower().count, "aeiou" + ("y" * include_y)))

# data
t = ibis.examples.penguins.fetch()
fuzzed = t.select(t.species, t.island).mutate(species_fuzzed=fuzz_str(t.species), island_fuzzed=fuzz_str(t.island)).select("species", "species_fuzzed", "island", "island_fuzzed")

species_choices = sorted(list(t.select("species").distinct().species.to_pandas()))
island_choices = sorted(list(t.select("island").distinct().island.to_pandas()))

species_num_vowel_map = {num_vowels(s): s.lower() for s in species_choices}

# llm nonsense
## #prompt-engineering -- need to pay for more courses to improve 1000x
prompt = f"""
Finish this prompt, outputting the best match.

1. Use the original choice string 
2. Output only that choice string
3. Do not include any other output

Input choices: from {species_choices}

Input: $INPUT

Output: """

## model pipeline
generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True)

## Ibis/DuckDB UDF
@udf.scalar.python
def llm_unfuzz(s: str) -> str:
    return generate_text(prompt.replace("$INPUT", s))[0]["generated_text"]

