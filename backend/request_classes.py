from pydantic import BaseModel


class SearchTerm(BaseModel):
    term: str

class ConfirmedValues(BaseModel):
    fields: dict

class StatusQuery(BaseModel):
    doc_name: str
