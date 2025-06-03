import json, asyncio
from typing import Optional
from .chatclient import BaseChatClient, OpenAIChatClient
from .base_agent import function_tool, FunctionTool
from .search_engine import BingSearch

def calculator() -> str:
    @function_tool
    def calculator(expression: str)->str:
        """Securely evaluates an arithmetic expression in python __builtins__ runtime.
        Parameters:
            expression (str): A string containing a single arithmetic expression valid in python.
        Returns:
            str: The the eval(...) result of the arithmetic expression or an error message.
        """
        try:
            # Evaluate with restricted built-ins for basic arithmetic.
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    return calculator
    
def search_from_web(llm_client:Optional[BaseChatClient]= None, top_k:Optional[int]=None)->FunctionTool:
    search_engine = BingSearch(llm_client= llm_client, top_k=top_k)
    @function_tool
    async def search_from_web(query: str)->str:
        """Search up-to-date or additional information from web.
        Parameters:
            - query (type:str, required): search term
        Returns:
            str: string of a list of search result pages, or an error message.
        """
        try:
            search_result = await search_engine.search(query)
            search_result = [page.model_dump() for page in search_result]
            return json.dumps(search_result, ensure_ascii=False)
        except Exception as e:
            return f'Failed to search from web. query:"{str(query)}", reason:"{str(e)}"'
    return search_from_web

def search_from_web_batch(llm_client:BaseChatClient|None = None, top_k:Optional[int]=None)->FunctionTool:
    search_engine = BingSearch(llm_client= llm_client, top_k=top_k)
    @function_tool
    async def search_from_web_batch(queries: list[str])->str:
        """Search up-to-date or additional information from web with a list of search queries in a batch.
        Parameters:
            - queries (type:list[str], required): a list of search queries
        Returns:
            str: string of a batched list of search result pages, or an error message.
        """
        try:
            tasks = [search_engine.search(query) for query in queries]
            search_results = await asyncio.gather(*tasks)
            results_batch = {}
            for i in range(len(queries)):
                results_batch[queries[i]] = [page.model_dump() for page in search_results[i]]
            return json.dumps(results_batch, ensure_ascii=False)
        except Exception as e:
            return f'Failed to search from web. query:"{str(queries)}", reason:"{str(e)}"'
    return search_from_web_batch
