import os, re, json, aiohttp
import aiofiles
from typing import Optional, Tuple
from enum import Enum
from pydantic import BaseModel
from urllib.parse import urlparse
from collections.abc import AsyncGenerator
from charset_normalizer import detect
from playwright.async_api import async_playwright
import trafilatura
from bs4 import BeautifulSoup
from .multitask import run_in_thread_pool
from .identity import unique_hash

# pip install libs
# TO-DO: change to class to host aiohttp.ClientSession
class Return(Enum):
    BYTES = 0 # bytes
    TEXT = 1 # plain text
    JSON = 2 # dict object
    STREAM = 3 # chunks of bytes

async def _read_stream(r_:aiohttp.ClientResponse, chunk_size_:int|None=1024*4)->AsyncGenerator[bytes]:
    async for data in r_.content.iter_chunked(chunk_size_):
        yield data
    
    await r_.__aexit__(None,None,None) # close the aiohttp connection cleanly

async def _get_return(r_:aiohttp.ClientResponse, return_:Return, chunk_size_:int|None=1024*4)->str|bytes|dict|AsyncGenerator[bytes]:
    match return_:
        case Return.TEXT:
            return await r_.text()
        case Return.JSON:
            return json.loads(await r_.text())
        case Return.STREAM:
            return _read_stream(r_=r_, chunk_size_=chunk_size_)
        case _:
            return await r_.read()

async def get(
        url_:str,
        session_:Optional[aiohttp.ClientSession] = None,
        headers_:dict|None=None,
        chunk_size_:int|None = 1024*4,
        timeout_:int | None = 30, # 30 seconds total timeout by default
        return_:Return|None=Return.BYTES)->str|bytes|dict|AsyncGenerator[bytes]:
    if session_ is None:
        async with aiohttp.ClientSession() as session:
            r = await session.get(
                url_, 
                headers=headers_,
                timeout=aiohttp.ClientTimeout(connect=timeout_, total=timeout_)) # set timeout to 10 seconds
            r.raise_for_status()
            result = await _get_return(r_ = r, return_=return_, chunk_size_=chunk_size_)
            if return_!=Return.STREAM:
                await r.__aexit__(None,None,None)
            return result
    else:
        r = await session_.get(
            url_, 
            headers=headers_,
            timeout=aiohttp.ClientTimeout(connect=timeout_, total=timeout_))
        r.raise_for_status()
        result = await _get_return(r_=r, return_=return_, chunk_size_=chunk_size_)
        if return_!=Return.STREAM:
            await r.__aexit__(None,None,None)
        return result

async def post(url_:str,
               session_:Optional[aiohttp.ClientSession] = None,
               headers_:dict|None=None,
               data_:object|None = None,
               chunk_size_:int|None = 1024*4,
               timeout_:int | None = 300, # 5 minutes total timeout by default
               return_:Return|None=Return.BYTES)->str|bytes|dict|AsyncGenerator[bytes]:
    if session_ is None:
        async with aiohttp.ClientSession() as session:
            r =  await session.post(url_, 
                                headers=headers_,
                                data=data_,
                                timeout=aiohttp.ClientTimeout(connect=10, total=timeout_)) # set timeout to 10 seconds
            r.raise_for_status()
            result = await _get_return(r_=r, return_=return_, chunk_size_=chunk_size_)
            if return_!=Return.STREAM:
                await r.__aexit__(None,None,None)
            return result
    else:
        r = await session_.post(url_, 
                                 headers=headers_,
                                 data=data_,
                                 timeout=aiohttp.ClientTimeout(connect=10, total=timeout_)) # set timeout to 10 seconds
        r.raise_for_status()
        result = await _get_return(r_=r, return_=return_, chunk_size_=chunk_size_)
        if return_!=Return.STREAM:
            await r.__aexit__(None,None,None)
        return result 

# Network onnection with context manager
class NetConnection:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    def get_session(self):
        return self.session

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get(self, 
                  url_:str,
                  headers_:dict|None=None,
                  chunk_size_:int|None = 1024*4, 
                  timeout_:int | None = 30, 
                  return_:Return|None=Return.BYTES)->str|bytes|dict|AsyncGenerator[bytes]:
        
        return await get(session_=self.session, 
                         url_=url_,
                         headers_=headers_,
                         chunk_size_=chunk_size_,
                         timeout_=timeout_,
                         return_=return_)
    
    async def post(self, 
                   url_:str,
                   headers_:dict|None=None,
                   data_:object|None = None,
                   chunk_size_:int|None = 1024*4, 
                   timeout_:int | None = 30, 
                   return_:Return|None=Return.BYTES)->str|bytes|dict|AsyncGenerator[bytes]:
        
        return await post(session_=self.session, 
                          url_=url_,
                          headers_=headers_,
                          data_ = data_,
                          chunk_size_=chunk_size_,
                          timeout_=timeout_,
                          return_=return_)

# fetch webpage with URL
## timeout: 30 seconds by default
## if make_clean_==False: return raw html
## if make_clean_==True: return simplifed xml with trafilatura
async def fetch_webpage_content(url_:str, timeout_:int|None=30, make_clean_:bool|None=False)->str:
    def extract_with_trafilatura(html_content_:str):
        xml_content = trafilatura.extract(
            html_content_, output_format="xml", include_tables=True, include_links=True, include_images=False, include_comments=False, with_metadata=False
        )
        return xml_content

    def extract_with_soup(html_content_:str):
        content = BeautifulSoup(html_content_, 'html.parser').text
        while content.find("\n\n")>=0: # remove uncessary empty lines
            content = content.replace("\n\n", "\n")
        return content

    def get_simplified_content(html_content_:str):
        # fallback: trafilatura->readability->soup
        return extract_with_soup(html_content_)
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)"
        }
        raw_content = await get(url_=url_, headers_=headers, return_=Return.BYTES, timeout_=timeout_)
        # handle non utf-8 encoding
        encoding = detect(raw_content).get('encoding') or "utf-8"
        page_content = raw_content.decode(encoding)
        if make_clean_:
            page_content = await run_in_thread_pool(get_simplified_content, page_content)
        return page_content
    except Exception as e:
        print(f'**Exception when fetch web content from url: {url_}, Exception: {str(e)}')
        return f'**Error**: failed to read web page content from url: {url_}, reason:{str(e)}'

# fetch webpage with playwright (this will not be blocked) and return cleaned html
## wait_selector_: css selector
## timeout_: 30 seconds timeout by default
async def fetch_webpage_content_playwright(url_, wait_selector_=None, timeout_=30, make_clean_=False):
    """
    Launch a headless browser, navigate to `url`, optionally wait for a selector
    or a load-state, then return the fully rendered HTML.
    """
    try:
        timeout_ = timeout_*1000 # change to milliseconds
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            # Navigate and wait until network is mostly quiet
            await page.goto(url_, timeout=timeout_)
            # Optionally wait for a specific element
            if wait_selector_:
                await page.wait_for_selector(wait_selector_, timeout=timeout_)
            content = await page.content()
            await browser.close()
        if make_clean_:
            return trafilatura.extract(
                content, output_format="xml", include_tables=True, include_links=True, with_metadata=False
            )
    except Exception as e:
        print(f"warning: failed to fetch webpage text. reason: {str(e)}")
        return ""

# fetch webpage and extract pure text within 30 seconds
async def fetch_webpage_text(url_:str, timeout_:int|None=30)->str:
    def _parse_html_text(html_:str):
        content = BeautifulSoup(html_, 'html.parser').text
        while content.find("\n\n")>=0: # remove uncessary empty lines
            content = content.replace("\n\n", "\n")
        return content
    try:
        content = await fetch_webpage_content(url_, timeout_=timeout_)
        content = await run_in_thread_pool(_parse_html_text, content)
        return content
    except Exception as e:
        print(f"warning: failed to fetch webpage text. reason: {str(e)}")
        return ""

# Fetch file content in bytes and return (success, meta, content)
class WebFileMeta(BaseModel):
    mime_type : Optional[str] = None # mime type: 'text/html', 'image/jpeg', 'audio/mpeg', 'application/json', 'application/pdf'
    ext : Optional[str] = None # file extension
    size : Optional[int] = 0 # file size
    filename: Optional[str] = None  # file name

_FILENAME_RE = re.compile(r'filename\*=.*?\'\'(?P<fname>[^;]+)|filename="?([^";]+)"?')

def infer_filename_from_url(url: str) -> str | None:
    path = urlparse(url).path
    if path and '/' in path:
        fname = path.rsplit('/', 1)[-1]
        if '.' in fname:
            return fname
    return None

async def fetch_webfile_content(
        url_:str,
        session_:aiohttp.ClientSession,
        max_size_:int | None = 100*1024*1024,
        stream_:bool|None=False,
        chunk_size_:int|None=10*1024*1024,
        metadata_only_:bool|None=False)->Tuple[bool, WebFileMeta, bytes|AsyncGenerator[bytes]|None]:
    try:
        r = await session_.get(url_, timeout=aiohttp.ClientTimeout(total=10)) # set timeout to 10 seconds
        r.raise_for_status()
        mime_type = r.headers.get('content-type','').lower().strip()
        ext = ''
        filename = ''
        if mime_type!="application/octet-stream":
            if ';' in mime_type: # sometimes 'text/html;characterset=utf-8'
                mime_type = mime_type.split(';')[0]
            ext = mime_type.split("/")[1] if "/" in mime_type else ''
            if '-' in ext:  # the extension part maybe 'x-pdf'
                ext = ext.split("-")[1]
        else:
            mime_type = None
            filename = infer_filename_from_url(url_)
            ext = filename.split('.')[-1] if '.' in filename else ''
                    
        content_size = int(r.headers.get('content-length', '0'))
        xsize = int(r.headers.get('x-filesize', '0')) # some protocal may use 'x-filesize' for file size
        file_size = min(content_size, xsize) if xsize else content_size

        # --- filename inference from content-disposition---
        if not filename:
            # Try Content-Disposition header
            cd = r.headers.get('content-disposition')
            if cd:
                m = _FILENAME_RE.search(cd)
                if m:
                    # pick the named group or the second group
                    filename = m.group('fname') or m.group(2)
                    if not ext:
                        ext = filename.split('.')[-1] if '.' in filename else ''

        # 3) If still no filename but we know an extension + default name
        if not filename and ext:
            filename = f"download_{unique_hash(5)}.{ext}"

        file_meta = WebFileMeta(mime_type=mime_type, ext = ext, size = file_size, filename=filename)
        if metadata_only_:
            return (True, file_meta, None)
                
        if max_size_ and file_size > max_size_:
            return (False, file_meta, None)
        content = await _get_return(r_=r, return_=Return.BYTES if not stream_ else Return.STREAM, chunk_size_=chunk_size_)
        return (True, file_meta, content)
    except Exception as e:
        return (False, None, None)

# save file from url and return (sucess, local filepath, meta)
async def download_webfile(
        url_:str,
        filedir_:Optional[str] = "./",
        filepath_:Optional[str] = None,
        max_size_:Optional[int] = 100*1024*1024,
        )->Tuple[bool, str, WebFileMeta]:
    async with aiohttp.ClientSession() as session:
        ok, meta, stream = await fetch_webfile_content(
            url_,
            session_=session,
            stream_=True, # enable streaming
            max_size_=max_size_ # no size limit
        )
        if not ok or meta is None:
            print(f"Failed to download: {url_}")
            return (False, None, None)
        # Determine local file path
        local_path = filepath_
        if not local_path:
            if filedir_ and not filedir_.endswith("\\") and not filedir_.endswith("/"):
                filedir_ += "/"
            filename = meta.filename or infer_filename_from_url(url_) or f"download_{unique_hash(5)}.{meta.ext}"
            local_path = os.path.join(filedir_, filename)
        try:
            async with aiofiles.open(local_path, "wb") as f:
                async for chunk in stream:
                    await f.write(chunk)
            return (True, local_path, meta)
        except Exception as e:
            print(f"** Caught exeception: {str(e)}")
            return (False, local_path, meta)

        




