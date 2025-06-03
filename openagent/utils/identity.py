import uuid
import hashlib

def unique_id()->str:
    return str(uuid.uuid4())

def unique_hash(length:int|None = 32)->len:
    u = uuid.uuid4()
    hash_full = hashlib.md5(str(u).encode()).hexdigest()
    return hash_full[:length]
    
