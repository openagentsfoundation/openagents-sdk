import asyncio, types, inspect
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, List
import regex as re
from pydantic import BaseModel

"""
Implementation class for OpenAI LLM model.
"""

__CTX_NAME__ = "context" # the conversation history
class ContextType(BaseModel):
    pass
    #messages: List[Dict[str,Any]] # conversation history

Callback = Callable[..., Any]

@dataclass
class FunctionTool:
    """A tool that wraps a function. In most cases, you should use  the `function_tool` helpers to
    create a FunctionTool, as they let you easily wrap a Python function.
    """
    name: Optional[str] = None
    """The name of the tool, as shown to the LLM. Generally the name of the function."""
    description: Optional[str] = None
    """A description of the tool, as shown to the LLM"""
    json_schema: Optional[dict[str, Any]] = None
    annotation: Optional[str] = None # arguments and return types
    is_async: Optional[bool] = False
    """The JSON schema for the tool's parameters."""
    callback: Optional[Callback] = None #callback
    input_arguments: Optional[Dict[str, str]] = None

    # call actual function
    def __call__(self, *args, **kwargs):
        return self.callback(*args, **kwargs)
    
    # support method in class which has 'self'
    def __get__(self, instance, owner):
        # descriptor protocol: bind to instance
        if instance is None:
            # accessed on the class, return the raw tool
            return self
        # bind self (the FunctionTool) to instance so that __call__
        # will receive instance as first argument
        return types.MethodType(self, instance)
    
    def has_context_argument(self)->bool:
        signature = inspect.signature(self.callback)
        result = False
        for param in signature.parameters.values():
            if param.name == __CTX_NAME__ and issubclass(param.annotation, ContextType):
                result = True
                break
        return result
#!Note "context" is reserved parameter by agent, it will be removed from json desc of the function
def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    if isinstance(func, FunctionTool):
        return func.json_schema

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    required = []

    signature_params = list(signature.parameters.values())
    if len(signature_params)>0 and signature_params[0].name in ("self", "cls"):
        signature_params = signature_params[1:] # ignore self and cls for class method
    
    for param in signature_params:
        try:
            if param.name == __CTX_NAME__ and issubclass(param.annotation, ContextType):
                continue
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}
        if param.default == inspect._empty:
            required.append(param.name)
    
    tool_js = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
    return tool_js

def transform_string_function_style(name: str) -> str:
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    return name.lower()

def function_tool(
    _func: Callback | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    )->FunctionTool:
    function_tool = FunctionTool(name=name, description=description)
    # THIS is the decorator that Python will call with your function
    def _decorate(callback:Callback):
        function_tool.name = function_tool.name if function_tool.name else callback.__name__ or ""
        function_tool.description = function_tool.description if function_tool.description else callback.__doc__ or ""
        function_tool.callback = callback     # store the real function
        function_tool.is_async = asyncio.iscoroutinefunction(callback)
        function_tool.json_schema = function_to_json(callback)
        function_tool.json_schema["function"]["name"] = function_tool.name
        function_tool.json_schema["function"]["description"] = function_tool.description

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }
        function_tool.input_arguments = {}
        signature = inspect.signature(callback)
        signature_params = list(signature.parameters.values())
        if len(signature_params)>0 and signature_params[0].name in ("self", "cls"):
            signature_params = signature_params[1:] # ignore self and cls for class method
        for param in signature_params:
            if param.name == __CTX_NAME__ and issubclass(param.annotation, ContextType):
                continue
            function_tool.input_arguments[param.name] = type_map.get(param.annotation, "string")
        
        return function_tool # replace `func` with your tool
    
    if callable(_func):
        return _decorate(_func)
    
    return _decorate            # give Python the decorator

def make_function_tool(
    name: str | None = None,
    description: str | None = None,
    callback:Callback | None = None)->FunctionTool:
    return function_tool(name=name, description=description)(callback)



