import asyncio
from concurrent.futures import ThreadPoolExecutor
__thread_pool_executor = ThreadPoolExecutor(max_workers=16) # to execute
__single_executor = ThreadPoolExecutor(max_workers=1) # to execute

def get_thread_pool():
    return __thread_pool_executor

# execute resource-consuming sync function in thread pool
async def run_in_thread_pool(function_, *args_):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_thread_pool(), function_, *args_)

async def run_in_single_worker(function_, *args_):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(__single_executor, function_, *args_)