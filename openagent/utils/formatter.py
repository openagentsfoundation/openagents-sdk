import json, re
from json.decoder import JSONDecodeError
from collections.abc import AsyncGenerator
from typing import Optional, Dict, Any
from pydantic import BaseModel

# replace template with variables by key '{{...}}' in dictionary
def format_template_with_json(template:str, variables:Dict[str, Any]):
    result = str(template)
    for key, value in variables.items():
        result = result.replace("{{" + key +"}}", str(value or ""))
    return result

# replace template with variables by key '{{...}}' in dictionary
def format_template_with_object(template:str, variables:BaseModel):
    variables_json = variables.model_dump()
    return format_template_with_json(template, variables_json)

# read a string word by word.
class AtomWordReader:
    """
    Read word by word from a string.  If a letter is immediately preceded
    by a backslash (e.g. "\n", "\t"), split it off as its own atom.
    Usage:
    reader = AtomWordReader()
    print(list(reader.emit_atoms("hello\\nworld")))
    # or
    for w in reader.emit_atoms("hello\\nworld"):
        print(w)
    """
    pattern = re.compile(r'''
        (?<=\\)[A-Za-z0-9_]    # a single word-char preceded by a backslash
        | [A-Za-z0-9_]+        # any run of word characters
        | \s                   # any single whitespace
        | [^\sA-Za-z0-9_]      # any other single non-word, non-space character
    ''', re.VERBOSE)

    def emit_atoms(self, text: str):
        for atom in self.pattern.findall(text):
            yield atom

"""
JsonStreamReader is to read first layer of json
"""
class JsonChunk(BaseModel):
    key: Optional[str]               # the key of the current field
    done: Optional[bool] = False     # whether the entire JSON object is done
    value_done: bool = False         # whether the field's value has been fully read
    value: Optional[Any] = ''        # chunk of decoded value for this field: a str, bool, list, dict, or None
    value_type: Optional[str] = None # one of 'string','integer','float','boolean','null','object','array'
    data: Dict[str, Any]             # the first-level object parsed so far

class JsonStreamReader:
    def __init__(self, stream: AsyncGenerator[str])->None:
        self.stream = stream

    async def read_one_layer(self) -> AsyncGenerator[JsonChunk]:
        # Parser states
        (WAIT_START, WAIT_KEY, READ_KEY, WAIT_COLON, WAIT_VALUE,
         IN_STRING, IN_ATOM, IN_OBJECT, IN_ARRAY) = range(9)

        state = WAIT_START
        key_buf = ''
        current_key: Optional[str] = None
        val_buf = ''
        val_type: Optional[str] = None
        depth = 0
        escape = False
        data: Dict[str, Any] = {}
        full_buffer = ''  # for error context

        # Mapping for JSON escape sequences
        escape_map = {
            'n': '\n',
            't': '\t',
            'r': '\r',
            '"': '"',
            '\\': '\\'
        }

        def emit_atom() -> JsonChunk:
            nonlocal val_buf, val_type, data, current_key, full_buffer
            try:
                # Use json.loads to parse number, boolean, or null
                parsed = json.loads(val_buf)
            except JSONDecodeError as e:
                raise JSONDecodeError(e.msg, full_buffer, len(full_buffer) - len(val_buf))

            # Determine the output type
            if parsed is None:
                out_type = 'null'
            elif isinstance(parsed, bool):
                out_type = 'boolean'
            elif isinstance(parsed, int):
                out_type = 'integer'
            elif isinstance(parsed, float):
                out_type = 'float'
            else:
                out_type = val_type

            data[current_key] = parsed
            chunk = JsonChunk(
                key=current_key,
                done=False,
                value_done=True,
                value=parsed,
                value_type=out_type,
                data=data.copy()
            )
            val_buf = ''
            return chunk

        async for tok in self.stream:
            full_buffer += tok

            if state == WAIT_START:
                if tok == '{':
                    state = WAIT_KEY
                else:
                    raise JSONDecodeError("Expecting '{' at start", full_buffer, len(full_buffer) - 1)
                continue

            if state == WAIT_KEY:
                if tok.isspace() or tok == ',':
                    continue
                if tok == '"':
                    key_buf = ''
                    state = READ_KEY
                elif tok == '}':
                    yield JsonChunk(key=None, done=True, data=data.copy())
                    return
                else:
                    raise JSONDecodeError("Expecting property name or '}'", full_buffer, len(full_buffer) - 1)
                continue

            if state == READ_KEY:
                if escape:
                    key_buf += tok
                    escape = False
                elif tok == '\\':
                    escape = True
                elif tok == '"':
                    current_key = key_buf
                    state = WAIT_COLON
                else:
                    key_buf += tok
                continue

            if state == WAIT_COLON:
                if tok == ':':
                    state = WAIT_VALUE
                elif not tok.isspace():
                    raise JSONDecodeError("Expecting ':' after key", full_buffer, len(full_buffer) - 1)
                continue

            if state == WAIT_VALUE:
                if tok.isspace():
                    continue
                if tok == '"':
                    val_buf = ''
                    escape = False
                    val_type = 'string'
                    state = IN_STRING
                elif tok == '-' or tok.lstrip('-').isdigit():
                    # Start of integer or float
                    val_buf = tok
                    val_type = 'number'
                    state = IN_ATOM
                elif tok in ('true', 'false', 'null'):
                    # Full boolean or null token
                    val_buf = tok
                    val_type = 'atom'
                    state = IN_ATOM
                elif tok == '{':
                    val_buf = tok
                    depth = 1
                    val_type = 'object'
                    state = IN_OBJECT
                    data[current_key] = val_buf
                    yield JsonChunk(key=current_key, done=False, value_done=False,
                                    value=tok, value_type='object', data=data.copy())
                elif tok == '[':
                    val_buf = tok
                    depth = 1
                    val_type = 'array'
                    state = IN_ARRAY
                    data[current_key] = val_buf
                    yield JsonChunk(key=current_key, done=False, value_done=False,
                                    value=tok, value_type='array', data=data.copy())
                else:
                    raise JSONDecodeError("Unexpected value token", full_buffer, len(full_buffer) - 1)
                continue

            if state == IN_STRING:
                if escape:
                    mapped = escape_map.get(tok, tok)
                    val_buf += mapped
                    data[current_key] = val_buf
                    yield JsonChunk(key=current_key, done=False, value_done=False,
                                    value=mapped, value_type='string', data=data.copy())
                    escape = False
                elif tok == '\\':
                    escape = True
                elif tok == '"':
                    data[current_key] = val_buf
                    yield JsonChunk(key=current_key, done=False, value_done=True,
                                    value=val_buf, value_type='string', data=data.copy())
                    state = WAIT_KEY
                    current_key = None
                else:
                    val_buf += tok
                    data[current_key] = val_buf
                    yield JsonChunk(key=current_key, done=False, value_done=False,
                                    value=tok, value_type='string', data=data.copy())
                continue

            if state == IN_ATOM:
                # Atom could be number, boolean, or null
                if tok in (',', '}', ' ', '\n', '\t'):
                    # finalize atom
                    yield emit_atom()
                    state = WAIT_KEY
                    current_key = None
                    if tok == '}':
                        yield JsonChunk(key=None, done=True, data=data.copy())
                        return
                else:
                    val_buf += tok
                continue

            if state == IN_OBJECT:
                if tok == '{' and not escape:
                    depth += 1
                elif tok == '}' and not escape:
                    depth -= 1
                val_buf += tok
                data[current_key] = val_buf
                # emit the closing brace char
                yield JsonChunk(key=current_key, done=False, value_done=False,
                                value=tok, value_type='object', data=data.copy())
                if depth == 0:
                    parsed = json.loads(val_buf)
                    data[current_key] = parsed
                    yield JsonChunk(key=current_key, done=False, value_done=True,
                                    value=parsed, value_type='object', data=data.copy())
                    state = WAIT_KEY
                    current_key = None
                continue

            if state == IN_ARRAY:
                if tok == '[' and not escape:
                    depth += 1
                elif tok == ']' and not escape:
                    depth -= 1
                val_buf += tok
                data[current_key] = val_buf
                # emit the closing bracket char
                yield JsonChunk(key=current_key, done=False, value_done=False,
                                value=tok, value_type='array', data=data.copy())
                if depth == 0:
                    parsed = json.loads(val_buf)
                    data[current_key] = parsed
                    yield JsonChunk(key=current_key, done=False, value_done=True,
                                    value=parsed, value_type='array', data=data.copy())
                    state = WAIT_KEY
                    current_key = None
                continue

        # end of stream
        print(full_buffer)
        raise JSONDecodeError("Unexpected end of JSON input", full_buffer, len(full_buffer))

