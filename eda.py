# imports
import os
import ibis
import torch
import random
import requests

import numpy as np
import pandas as pd

from ibis import _
from dotenv import load_dotenv
from transformers import pipeline
from ibis.expr.operations import udf

# load env variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

# openai settings
url = "https://api.openai.com/v1/chat/completions"
model = "gpt-3.5-turbo"

# configure ibis
ibis.options.interactive = True

