# env.py 
import os
from dotenv import load_dotenv
__dotenv_loaded = False
__dotenv_path = os.path.join(os.getcwd(), ".env")

def get(name_:str, default_=None):
    global __dotenv_loaded, __dotenv_path
    if not __dotenv_loaded:
        load_dotenv(dotenv_path=__dotenv_path)
        __dotenv_loaded = True
        print(f"Loaded .env at path: {__dotenv_path}")
    return os.getenv(name_, default_)
