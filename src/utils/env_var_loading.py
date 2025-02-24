from dotenv import load_dotenv
from src.common_entities.custom_exceptions import CustomException
import os

load_dotenv()


class DotEnvException(CustomException):
    pass


def get_env_var(name: str) -> str:
    env_var = os.getenv(name)

    if env_var is None:
        raise DotEnvException(f"Variable {name} cannot be loaded.")

    return env_var
