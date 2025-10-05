
import os
from typing import Union
from langgraph.pregel.io import AddableValuesDict

def _langgraph_message_to_mlflow_message(
    langgraph_message: AddableValuesDict,
) -> dict:
    langgraph_type_to_mlflow_role = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }

    if type_clean := langgraph_type_to_mlflow_role.get(langgraph_message.type):
        return {"role": type_clean, "content": langgraph_message.content}
    else:
        raise ValueError(f"Incorrect role specified: {langgraph_message.type}")


def get_most_recent_message(response: AddableValuesDict) -> dict:
    most_recent_message = response.get("messages")[-1]
    return _langgraph_message_to_mlflow_message(most_recent_message)["content"]


def increment_message_history(
    response: AddableValuesDict, new_message: Union[dict, AddableValuesDict]
) -> list[dict]:
    if isinstance(new_message, AddableValuesDict):
        new_message = _langgraph_message_to_mlflow_message(new_message)

    message_history = [
        _langgraph_message_to_mlflow_message(message)
        for message in response.get("messages")
    ]

    return message_history + [new_message]
