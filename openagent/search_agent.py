# Agentic Search Agent help search from web in a agent loop and return summarized proof to user query
from .utils import env as env
import json
from typing import Optional
# OpenAI chat model
from .base_agent import (
    FunctionTool,
    function_tool,
    MAX_STEPS,
    BaseChatClient
)
from .react import ReactAgent, MAX_STEPS
from .common_tools import search_from_web, search_from_web_batch
from .search_engine import BingSearch

"""
returns: {title, URL, content snippet, raw_content, answer, images}
"""

SEARCH_INSTRUCTIONS = """You are a ReAct Agent equipped with web search capabilities.

## Workflow:
Follow below steps strictly in your ReAct **reasoning loop**:
1. **Understand the Query**: Carefully analyze the user's question or problem to identify the key information needed.
2. **Plan the Search**: In your initial **though** when receiving user query, follow below rules to extend queries:
    - If user query has multiple atomic queries, breakdown into sub queries.
    - Expand more queries around the topic of user query, to fetch more information to answer.
    - Write your plan in descriptive manner, such as "To address the query about ..., I need search with following queries '[query 1]', '[query 2]' ..."
3. **Perform Web Searches**: Use the `search_from_web` tool to gather information (search result pages). Conduct multiple searches if necessary to refine or expand the results.
4. **Reflect Search Results** (In you reflection **thought**):
     (You received a list of search result pages of query)
   - Evaluate the information retrieved from the search results for accuracy, relevance, and completeness. 
   - Propose **new search queries** in your **thought** and ask yourself to search more results, if retrieved search results cannot address user's question.
   - **Don't** conclude to answer if the search results don't have **concrete proof** or **lack of details** to address user's query.
6. **Summarization**: (when you are comfortable to conclude an answer, provide comprehensive report to the user)
   - Ensure the information retrieved is from credible and authoritative sources.
   - Exclude outdated (if the query is time sensitive) search results or irrelevant information in the response.
     (Due to timezone difference, the query for **today** can be within 24 hours difference from the search result pages timestamp.)
   - Ensuring the **final_answer** is relavant, informative and comprehensive to user's query.
   - Ground your answer in the search result pages - do not invent or assume facts.
   - Output the **final_answer** value in format following the `## Report Format` section.

## Report Format:
1. Output the **final_answer** including content of:
- Comprehensive **Summary** including contents: 1. answer directly addressing related the user's query 2. informative content related to the query for extended reading.
- **Data tables**(in markdown) if provided in reseach results and related to user's query
- **Images**(in markdown) if provided in reseach results and related to user's query 
- **References Links**(in markdown): extracted from `url` from search results, where the answer is grounded.

You should rename the section name in the report in more human readable names according to the content of each section.

2. Note: the output **must** be in **language** of the query and in **Markdown**.
"""

class SearchAgent(ReactAgent):
    def __init__(
        self,
        llm_client:BaseChatClient, # assistant LLM
        name:Optional[str] = "WebSearcher",
        instructions:Optional[str] = SEARCH_INSTRUCTIONS,
        tools:list[FunctionTool]|None = [], # extra tools
        max_steps:int | None = MAX_STEPS,
        logger = None,
        verbose:bool|None=True):

        super().__init__(
            llm_client = llm_client,
            name = name,
            instructions = instructions,
            tools = tools,
            max_steps = max_steps,
            logger = logger,
            verbose = verbose)
        
        self.search_engine = BingSearch(market="zh-CN", top_k=5)
        self.register_tools([search_from_web(llm_client)]) # register search tools
    
    # search from web and returns a list of search result pages
    # result page: {publish_time, title, snippet, content}
    @function_tool
    async def search_from_web(self, query: str)->str:
        """Search up-to-date or additional information from web.
        Parameters:
            - query (type:str, required): search term
        Returns:
            str: string of a list of search result summary, or an error message.
        """
        try:
            search_result = await self.search_engine.search(query)
            return json.dumps(search_result, ensure_ascii=False)
        except Exception as e:
            return f'Failed to search from web. query:"{str(query)}", reason:"{str(e)}"'
    

    