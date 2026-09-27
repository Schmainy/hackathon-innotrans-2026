"""Map the variable names from .env.example onto the names used in the code.

.env.example uses the AZURE_OPENAI_* names; the code reads AZURE_API_KEY, AZURE_ENDPOINT
and MODEL_NAME / AZURE_MODEL. Both spellings work; a name that is already set wins.
"""
import os

ALIASES = {
    "AZURE_API_KEY": "AZURE_OPENAI_API_KEY",
    "AZURE_ENDPOINT": "AZURE_OPENAI_ENDPOINT",
    "MODEL_NAME": "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_MODEL": "AZURE_OPENAI_DEPLOYMENT",
}


def apply_env_aliases() -> None:
    """Copy AZURE_OPENAI_* values to the names the code reads, unless those are already set."""
    for target, source in ALIASES.items():
        if not os.environ.get(target) and os.environ.get(source):
            os.environ[target] = os.environ[source]
