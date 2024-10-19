from typing import List

from pydantic import BaseModel


class ModelKWArgs(BaseModel):
    modelParameter: dict = {
        "temperature": 0.75,
        "max_tokens": 2000,
        "top_p": 0.9,
    }


class RequestModel(ModelKWArgs):
    userID: str
    requestID: str
    user_input: str


class MermaidRequest(BaseModel):
    userID: str
    requestID: str


class RequestModelProject(ModelKWArgs):
    id: str
    text: str
    response: str


class UserRequest(ModelKWArgs):
    userID: str
    requestID: str
    user_input: List[RequestModelProject]
