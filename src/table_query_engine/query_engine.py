import os
import pandas as pd
from .models import QueryResponse

from llama_index.core.query_engine import NLSQLTableQueryEngine
from vllm import LLM, SamplingParams

class QueryEngine:
    def __init__(self, llm, df):
        self.query_engine = NLSQLTableQueryEngine(sql_database=df, llm=llm, verbose=True)

    def __call__(self, query_str) -> QueryResponse:
        response = self.query_engine.query(query_str)
        return QueryResponse(response=response.response)


def initialize_query_engine():
    
    sampling_params = SamplingParams(use_beam_search = True , early_stopping = True , best_of = 5)

    # Create an LLM.
    llm = LLM(model="facebook/opt-125m" ,)

    return QueryEngine(llm, df)
