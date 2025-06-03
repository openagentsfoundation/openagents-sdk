"""
All exceptions thrown by agents sdk
"""
from enum import Enum
from typing import Optional
import json

class ErrorCode(Enum):
    NoError = 0 # no error found
    Unknown = 1 # unknown error, it's caused by unexpected exception
    LlmAccessError = 2 # Llm not connected, unauthenticated, reached rate limit or any access related error. 
    ModelBehaviorError = 3 # Agent cannot run correctly as expected, due to issues such as unexisting tools/mcps/resources, llm json output error
    InvalidInput = 4 # Input validation failed, by user defined user input validator
    InvalidOutput = 5 # Output validation failed, by user defined output validator

class AgentException(Exception):
    """Base class for all exceptions in the Agents SDK."""
    def __init__(self, details:Optional[str]=None, error_code:Optional[ErrorCode]=ErrorCode.Unknown, module:Optional[str]=None) -> None:
        """
        Args:
        - details (str): the detailed description of the exception
        - error_code: the error code defined above
        - module: the module or agent which triggered the exception, it's for logging only.
        """
        module = module or "" # set to valid string
        if details is None:
            details = json.dumps({"exception":self.__class__.__name__, "module":module, "error_code":str(error_code)})
        super().__init__(details)
        self.module:Optional[str] = module
        self.error_code:Optional[ErrorCode] = error_code # for seraiable
