from .utils import env
from pydantic import BaseModel
from typing import Optional, Literal, Any, Dict
from pprint import pprint
import json, urllib.parse, asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from abc import ABC, abstractmethod
from .utils import net
from .chatclient import BaseChatClient, OpenAIChatClient, Prompt, Options
# TO-DO: add cache for url->webpage
# Search and summize with GPT
## @ return summary and references websites.
## client side and render a screenshot of website with html2canvas: https://stackoverflow.com/questions/4912092/using-html5-canvas-javascript-to-take-in-browser-screenshots/6678156#6678156
SEARCH_RESULTS_SUMMARY_SYSTEM_PROMPT = \
"""You are expert in summarize search results.
**TIME NOW** is {{date_time}}.
You will receive a list of search results in json array, for search query: "{{search_query}}".
Your task is to convert the received content into json in following format and output without ```JSON mark:
'''
{
  "summary":"..." # comphrehensive summary of search result pages, addressing search query.
  "references": [...,...] # a list of int indexes (seperated by ',') of search result pages, whose content are referenced in your summary.
}
'''

# Rules
Your **must** follow the below rules step by step to generate the response:
1. Exclude search results not related to search query
2. If search query requires freshness of information, exclude non-freshed search results and only keep the latest, by comparing timestamp of search results with time now.
3. After finish of above 2 steps, put the indexes (integer) of remaining search results to references list. 
4. Summarize each search result to sub-summary for remaining search results.
5. Summarize sub-summaries of previous step to final summary in less than 256 words in **language** of search query.

# Output Format
output in json.
"""

SEARCH_RESULTS_FILTER_INSTRUCTIONS = """You are an expert in ranking and filtering Bing search results.  

**CURRENT TIME**: {{date_time}}  
**SEARCH QUERY**: "{{search_query}}"  
**MAX RESULTS TO RETURN**: {{top_k}}

You will be given a JSON-lines list of search results. Each result has these fields:
```json
{
  "index":  <int>,           # the original position in the result list
  "url":    "<str>",         # the page URL
  "timestamp": "<str>",      # the page's published datetime, ISO-format
  "title":  "<str>",         # the page title
  "snippet":"<str>"          # the snippet text around the query terms
}

Your task is to select the up-to **{{top_k}}** most relevant results and return their indices, ordered by **descending relevance**.

## 1. Detecting Time Sensitivity
A query is time-sensitive if it explicitly or implicitly requests recency (e.g. “recent,” “latest,” “today,” “breaking,” “news,”, etc.).

- If the query is time-sensitive:
  1. Exclude any result older than 1 year (unless otherwise specified).
  2. Rank primarily by timestamp (newest first), then by topical relevance. 

- If the query is **not** time-sensitive, ignore the timestamp except to break ties between equally relevant items (newer → higher).

## 2. Topical Relevance Criteria
For every candidate, judge how well the **title** and **snippet** address the query. Consider:
- **Exact match** of key terms or entities.
- **Semantic relevance**: Is the page about the query subject?
- **Authority cues**: known reputable domains or publication names in URL/title.
Assign each result a combined **time + topical** relevance score (higher = better).

## 3. Threshold & Selection
- You may return **fewer** than {{top_k}} if some slots would fall below your relevance threshold.
- Only include results you judge *truly* relevant.
- You need consider **timezone difference**, for example: **today** may have 24 hours timezone difference from the time stamp of search result page.
- Sort your final list by descending relevance.

## 4. Output
Return *only* this JSON object (no extra commentary):
```json
{
  "relevant_indexes": [<int>, <int>, …]
}

- up to {{top_k}} values, in order of relevance.
"""

SUMMARIZE_PAGE_INSTRUCTIONS = """You are a research assistant. You will receive:

  1. An HTML/XML/Plain text fragment or of sanitized web-page content.  
  2. The user's original **search query**: "{{search_query}}".

Your job is to output valid **Markdown** with exactly three sections—each including *only* items that are directly **relevant** to the user’s query.
You will return:
- **Summary**: the comphrehensive overview (don't miss key data points) of content related to search query.
- **Data Table**: the data supports the summary and related to the search query.
- **Followup Links**: the links which is related to the search query for further reading.
- **Don't** make-up any facts not in the provided fragment

The details are below:

### 1. Summary  
A comphrehensive overview of the fragment, emphasizing any parts that address the user’s query. Omit any details unrelated to the query.

### 2. Data Tables: 
Scan the XML for data tables - any `<table>` elements whose **row contents** or **column headers** match or speak to the search query’s topic or keywords.  
- **Include** only those tables.
- Render each in Markdown:
  ```markdown
  | Header1 | Header2 | … |
  | ------- | ------- | - |
  | …       | …       |   |

- If **no data table** is relevant, this section should say "No relevant data tables found."

### 3. Followup Links
- Identify all link tags that:
  1. Link to the same domain as the fragment.
  2. Have a path-depth ≥ 2 (e.g. example.com/section/page).
  3. Whose anchor text or surrounding context indicates relevance to the search query.

- List each as a Markdown bullet:
  ```markdown
  - [Anchor Text](https://…)

- If none match, say "No relevant followup links found."

Output must be valid Markdown, structured exactly with these three headings, and in the same language as the page content.
"""

BING_SEARCH_ENDPOINT =  env.get('BING_SEARCH_ENDPOINT')
BING_SEARCH_API_KEY =  env.get('BING_SEARCH_API_KEY')
BING_SEARCH_MARKET = env.get('BING_SEARCH_MARKET', "en-US") #addition configs

GOOGLE_SEARCH_ENDPOINT =  env.get('GOOGLE_SEARCH_ENDPOINT')
GOOGLE_SEARCH_API_KEY =  env.get('GOOGLE_SEARCH_API_KEY')
GOOGLE_SEARCH_MARKET = env.get('GOOGLE_SEARCH_MARKET', "en-US") #addition configs

SEARCH_RESULT_TOP_K = int(env.get('SEARCH_RESULT_TOP_K', "5"))
MAX_PAGE_TEXT_SIZE = 4*1024

class SearchResultSummary(BaseModel):
    summary: Optional[str] = None
    references: Optional[list[dict]] = None # list of search result page in json

class SearchResultPage(BaseModel):
    url: Optional[str] = None
    timestamp: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    content: Optional[str] = None

SearchMode = Literal["search", "research"]
class BaseSearchEngine(ABC):
    def __init__(
        self,
        llm_client:Optional[BaseChatClient] = None, # LLM used for summarizer
        end_point:Optional[str]=None,
        api_key:Optional[str]=None,
        market:Optional[str]=None,
        top_k:Optional[int]=None,
        mode:Optional[SearchMode] = "search"):
        
        self.llm_client = llm_client or OpenAIChatClient()
        self.end_point = end_point
        self.api_key = api_key
        self.market = market
        self.top_k = top_k
        self.mode = mode
        self.cache:Optional[Dict[str, str]] = {}

    def cache_webpage(self, url:str, page_content:str):
        self.cache[url] = page_content

    def get_cached_webpage(self, url:str):
        return self.cache.get(url)
    # Search from web and returns top_k results of SearchResultPage in a list
    # SearchResultPage: {
    #   url:str # URL of the the result page
    #   timestamp:str # the published datetime (local timezone) of the result page
    #   title: str # the title of the the result page
    #   snippet: str # snippet from search api
    #   content: str: # summary of the result page. empty if exception
    # }
    @abstractmethod
    async def search(self, query:str)->list[SearchResultPage]:
        pass
    
    # help method to re-rank and select (by LLM) top-k relevant pages
    async def _select_top_k_relevant(self, result_pages:list[SearchResultPage], query:str, top_k:int)->list[SearchResultPage]:
        system_prompt = SEARCH_RESULTS_FILTER_INSTRUCTIONS.replace("{{search_query}}", query)
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        system_prompt = system_prompt.replace("{{date_time}}", now_str)
        system_prompt = system_prompt.replace("{{top_k}}", str(top_k))
        result_pages_jsonl = [page.model_dump(exclude_unset=True, exclude_none=True) for page in result_pages]
        for i in range(len(result_pages_jsonl)):
            result_pages_jsonl[i]["index"] = i
            # change to local time zone
            utc_datetime = datetime.strptime(result_pages_jsonl[i]["timestamp"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=ZoneInfo("UTC"))
            local_timezone = datetime.now().astimezone().tzinfo
            local_datetime = utc_datetime.astimezone(local_timezone)
            result_pages_jsonl[i]["timestamp"] = local_datetime.strftime('%Y-%m-%d %H:%M:%S')

        user_prompt = json.dumps(result_pages_jsonl, ensure_ascii=False) # support Chinese characters
        response = await self.llm_client.send(
                prompt=user_prompt,
                system=system_prompt,
                options=Options(response_format={ "type": "json_object"}),
                stream=False)
        js_result = json.loads(response.text) if response.text else {}
        indexes = js_result.get("relevant_indexes") or []
        return [result_pages[i] for i in indexes if i>=0 and i<len(result_pages)]

    # Read webpage and summarize in markdown
    ## return empty string is any exception caught
    async def fetch_webpage_content(self, query:str, url:str)->str:
        print(f"call fetch_webpage_content. query:{query}, url:{url}")
        if self.mode == "research": # fetch more detail, including data table, deep links
            cleaned_html = await net.fetch_webpage_content_playwright(url_=url, timeout_=30, make_clean_=True)
            if cleaned_html:
                user_prompt = f"Web Page Content in XML:\n'''\n{cleaned_html}\n'''"
                system_prompt = SUMMARIZE_PAGE_INSTRUCTIONS.replace("{{search_query}}", query)
                response = await self.llm_client.send(prompt=user_prompt, system = system_prompt, stream=False)
                return response.text
            else:
                return ""
        else:
            text_content = await net.fetch_webpage_text(url_=url, timeout_=10) # fetch only plain text
            if text_content:
                user_prompt = f"Web Page Content in Plain Text:\n'''\n{text_content}\n'''"
                system_prompt = SUMMARIZE_PAGE_INSTRUCTIONS.replace("{{search_query}}", query)
                response = await self.llm_client.send(prompt=user_prompt, system = system_prompt, stream=False)
                return response.text
            else:
                return ''
    # read file content and convert to markdown
    async def fetch_file_content(self, filepath_url:str):
        """Fetch file page via full file path (web url) and return markdown formmatted file content"""
        pass

    # fetch multiple webpages at once, set 10 seconds for timeout
    async def fetch_webpage_batch(self, query:str, urls:list[str])->list[str]:
        """Fetch file page via file URLs in a batch"""
        tasks = [self.fetch_webpage_content(query=query, url=url) for url in urls]
        return await asyncio.gather(*tasks)

    async def summarize_search_results(self, query:str, result_pages:list[SearchResultPage])->SearchResultSummary:
        system_prompt = SEARCH_RESULTS_SUMMARY_SYSTEM_PROMPT.replace("{{search_query}}", query)
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        system_prompt = system_prompt.replace("{{date_time}}", now_str)
        result_pages_jsonl = [page.model_dump() for page in result_pages]
        for i in range(len(result_pages_jsonl)):
            result_pages_jsonl[i]["index"] = i
            # change to local time zone
            utc_datetime = datetime.strptime(result_pages_jsonl[i]["timestamp"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=ZoneInfo("UTC"))
            local_timezone = datetime.now().astimezone().tzinfo
            local_datetime = utc_datetime.astimezone(local_timezone)
            result_pages_jsonl[i]["timestamp"] = local_datetime.strftime('%Y-%m-%d %H:%M:%S')

        user_prompt = json.dumps(result_pages_jsonl)
        try:
            prompt = Prompt(text=user_prompt)
            response = await self.llm_client.send(
                prompt=prompt,
                system=system_prompt,
                options=Options(response_format={ "type": "json_object"}),
                stream=False)
            
            if response.text:
                js_result = json.loads(response.text) if response.text else {}
                summary = SearchResultSummary(summary = js_result.get("summary") or "", references=[])
                indexes = js_result.get("references") or []
                if indexes:
                    summary.references = [result_pages[int(index)].model_dump(exclude={"content"}) for index in indexes]
                return summary
        except Exception as e:
            print(f'exception when call summarize_search_results. reason: {str(e)}')

        return SearchResultSummary(summary = '', references=[])
    
    async def search_return_summary(self, query:str)->SearchResultSummary:
        result_pages = await self.search(query)
        summary = await self.summarize_search_results(query, result_pages)
        return summary

class BingSearch(BaseSearchEngine):
    def __init__(
        self,
        llm_client:Optional[BaseChatClient] = None,
        end_point:Optional[str]=None,
        api_key:Optional[str]=None,
        market:Optional[str]=None,
        top_k:Optional[int]=None,
        mode:Optional[SearchMode] = "search"):
        end_point = end_point or BING_SEARCH_ENDPOINT
        api_key = api_key or BING_SEARCH_API_KEY
        market = market or BING_SEARCH_MARKET or "en-US"
        top_k = top_k or SEARCH_RESULT_TOP_K
        super().__init__(
            llm_client = llm_client,
            end_point = end_point,
            api_key = api_key,
            market = market,
            top_k = top_k,
            mode = mode,
        )
    
    # implementation of bing search
    async def search(self, query:str)->list[SearchResultPage]:
        headers={'Ocp-Apim-Subscription-Key':self.api_key}
        q = urllib.parse.quote(query)
        request_url = f'{self.end_point}?q={q}&mkt={self.market}&customconfig=0&count={self.top_k*2}'
        try:
            js_result = await net.get(url_=request_url, headers_=headers, return_=net.Return.JSON, timeout_=30)
        except Exception as e:
            raise Exception("Search engine API access failed")
        
        if (js_result.get("error") or not js_result.get("webPages")):
            return []
        returned_pages:list[SearchResultPage] = []
        index = 0
        for page in js_result["webPages"]["value"]: # top k pages
            page = dict(page)
            returned_pages.append(SearchResultPage(
                url=page["url"],
                timestamp = page.get("datePublished", page.get("dateLastCrawled"))[0:19],
                title=page["name"],
                snippet=page["snippet"]))
            index+=1
        ranked_pages = await self._select_top_k_relevant(result_pages=returned_pages, query=query, top_k=self.top_k)
        #print("**ranked pages**")
        #pprint(ranked_pages)
        
        fetched_contents = await self.fetch_webpage_batch(query=query, urls=[page.url for page in ranked_pages])
        #print("**fetched_contents**")
        #pprint(fetched_contents)
        for i in range(len(ranked_pages)):
            ranked_pages[i].content = fetched_contents[i] if len(fetched_contents[i]) < MAX_PAGE_TEXT_SIZE else fetched_contents[i][0:MAX_PAGE_TEXT_SIZE]
        return ranked_pages
        
class GoogleSearch(BaseSearchEngine):
    def __init__(
        self,
        llm_client:Optional[BaseChatClient] = None,
        end_point:Optional[str]=None,
        api_key:Optional[str]=None,
        market:Optional[str]=None,
        top_k:Optional[int]=None):
        end_point = end_point or GOOGLE_SEARCH_ENDPOINT
        api_key = api_key or GOOGLE_SEARCH_API_KEY
        market = market or GOOGLE_SEARCH_MARKET or "en-US"
        top_k = top_k or SEARCH_RESULT_TOP_K
        super().__init__(
            llm_client = llm_client,
            end_point = end_point,
            api_key = api_key,
            market = market,
            top_k = top_k
        )
    
    # implementation of bing search
    async def search(self, query:str)->list[SearchResultPage]:
        pass