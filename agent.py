import os
from google.adk.agents import LlmAgent, Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
from google.genai import types
from typing import List

# --------------------------------------------------------------------------
# 1. INFRASTRUCTURE SETUP (API Key Fix)
# The deployment environment sets GOOGLE_API_KEY. We must read it here.
# --------------------------------------------------------------------------
API_KEY_FROM_ENV = os.environ.get("GOOGLE_API_KEY")

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# --------------------------------------------------------------------------
# 2. DAY 4 TOOL FIX: count_papers (Corrected for deployment)
# This tool now accepts the correct list type (List[str]).
# --------------------------------------------------------------------------
def count_papers(papers: List[str]) -> int:
    """
    This function counts the number of papers in a list of strings (research papers).
    Args:
      papers: A list of strings, where each string is a research paper result.
    Returns:
      The number of papers in the list.
    """
    # Assuming search results are returned as a list of distinct items
    return len(papers)


# --------------------------------------------------------------------------
# 3. DAY 4 AGENT DEFINITION (Delegation Example)
# --------------------------------------------------------------------------

# Google Search agent (Specialist Agent)
google_search_agent = LlmAgent(
    name="google_search_agent",
    # Pass the API key explicitly
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, api_key=API_KEY_FROM_ENV),
    description="Searches for information using Google search",
    instruction="""Use the google_search tool to find information on the given topic. Return the raw search results as a list of strings.""",
    tools=[google_search]
)

# Root agent (The main agent accessible via the UI)
research_paper_finder_agent = LlmAgent(
    name="research_paper_finder_agent",
    # Pass the API key explicitly
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, api_key=API_KEY_FROM_ENV),
    instruction="""Your task is to find research papers and count them. 
    You MUST ALWAYS follow these steps:
    1) Find research papers on the user provided topic using the 'google_search_agent'. 
    2) Then, pass the papers to 'count_papers' tool to count the number of papers returned.
    3) Return both the list of research papers and the total number of papers.""",
    # Uses delegation (Day 4) and a custom tool
    tools=[AgentTool(agent=google_search_agent), count_papers]
)

# --------------------------------------------------------------------------
# 4. EXPOSE AGENTS (Fix for the deployment UI/ADK Web Server)
# --------------------------------------------------------------------------
# The ADK Web Server looks for all Agent definitions in the file. 
# We explicitly define the root agent as the final output.

root_agent = research_paper_finder_agent
