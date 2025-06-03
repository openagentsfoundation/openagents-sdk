import base64
from pydantic import BaseModel
from typing import Optional
from pprint import pprint
from mimetypes import guess_type
from .net import fetch_webfile_content, WebFileMeta
    
async def _image_url_to_b64url(url_:str)->str:
    success, meta, content = await fetch_webfile_content(url_)
    if not success:
        return None
    b64_data = base64.b64encode(content).decode('utf-8')
    return f"data:{meta.mime_type};base64,{b64_data}"

def _image_local_to_b64url(path_:str)->str:
    mime_type, _ = guess_type(path_)
    if mime_type is None:
        mime_type = 'image/jpeg'  # Default MIME type if none is found
    try:
        with open(path_, "rb") as image_file:
            b64_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{b64_data}"
    except:
        return None

async def image_path_to_b64url(path_:str)->str:
    if path_ and path_.lower().startswith("http"):
        return await _image_url_to_b64url(path_)
    elif path_:
        return _image_local_to_b64url(path_)
    else:
        return None
