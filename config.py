import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Job search configuration
def _get_domains_from_env(env_var: str, default: list = None):
    """Helper function to parse comma-separated domains from environment"""
    env_domains = os.getenv(env_var)
    if env_domains:
        return [domain.strip() for domain in env_domains.split(",")]
    return default or []

# Include domains (defaults to LinkedIn and Indeed)
INCLUDE_DOMAINS = _get_domains_from_env("JOB_SEARCH_INCLUDE_DOMAINS", ["linkedin.com", "indeed.com"])

# Exclude domains
EXCLUDE_DOMAINS = _get_domains_from_env("JOB_SEARCH_EXCLUDE_DOMAINS")

# Default search parameters
DEFAULT_MAX_RESULTS = int(os.getenv("JOB_SEARCH_MAX_RESULTS", "10"))
DEFAULT_DAYS = int(os.getenv("JOB_SEARCH_DAYS", "7"))

# API keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
