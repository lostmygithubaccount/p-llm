# imports
import time
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

# configure torch
device = 0 if torch.cuda.is_available() else -1

# data
imdb_titles = ibis.examples.imdb_title_basics.fetch()
imdb_ratings = ibis.examples.imdb_title_ratings.fetch()

joined = imdb_titles.join(imdb_ratings, "tconst")

t = joined.relabel("snake_case")
t = t.filter((_.title_type == "movie") & (_.num_votes >= 50_000))
t = t.select("tconst", "primary_title", "average_rating", "num_votes")
t = t.order_by(_.average_rating.desc())
t = t.limit(1000)

## model pipeline
generate_text = pipeline(
    model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16, trust_remote_code=True
)

## Ibis/DuckDB UDF
@udf.scalar.python
def llm_describe(s: str) -> str:
    prompt = "Describe the movie: '{}'".format(s)
    return str(generate_text(prompt)[0]["generated_text"])

