from .utils import env

OPENAGENT_LLM_ENDPOINT=env.get("OPENAGENT_OPENAI_LLM_ENDPOINT")
OPENAGENT_LLM_API_KEY = env.get("OPENAGENT_OPENAI_LLM_API_KEY")
OPENAGENT_LLM_DEPLOYMENT = env.get("OPENAGENT_OPENAI_LLM_DEPLOYMENT")
OPENAGENT_LLM_API_VERSION = env.get("OPENAGENT_OPENAI_LLM_API_VERSION")
OPENAGENT_LLM_TEMPERATURE = env.get("OPENAGENT_OPENAI_LLM_TEMPERATURE", None)
OPENAGENT_LLM_TOP_P = env.get("OPENAGENT_OPENAI_LLM_TOP_P", None)
OPENAGENT_LLM_IS_REASONING = bool(env.get("OPENAGENT_OPENAI_LLM_IS_REASONING", "false").lower() == "true")
OPENAGENT_LLM_REASONING_EFFORT = env.get("OPENAGENT_OPENAI_LLM_REASONING_EFFORT", None)
OPENAGENT_MAX_STEPS = 30
