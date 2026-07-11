"""
Request schema — the shape of a valid POST /analyze body.

These Pydantic models define our API contract. FastAPI validates incoming JSON
against them automatically: anything that doesn't fit is rejected with a 422
before our code runs. Almost everything is optional with sensible defaults, so
callers can send partial emails without breaking validation.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime, timezone


class Sender(BaseModel):
    name: Optional[str] = ""
    email: Optional[EmailStr] = None      # EmailStr enforces valid email syntax
    domain: Optional[str] = ""            # the domain the sender claims to be from

    class Config:
        extra = "ignore"                  # silently drop unknown fields


class Receiver(BaseModel):
    email: Optional[EmailStr] = None

    class Config:
        extra = "ignore"


class Content(BaseModel):
    text: str = Field(default="", min_length=0)   # plain-text body
    html: Optional[str] = ""                       # HTML body (cleaned before use)

    class Config:
        extra = "ignore"


class Link(BaseModel):
    url: str
    domain: Optional[str] = ""            # used by the risk engine's link checks
    anchor_text: Optional[str] = ""

    class Config:
        extra = "ignore"


class Image(BaseModel):
    src: str
    alt: Optional[str] = ""

    class Config:
        extra = "ignore"


class Attachment(BaseModel):
    filename: str
    type: Optional[str] = ""

    class Config:
        extra = "ignore"


class Metadata(BaseModel):
    num_links: int = 0
    num_images: int = 0
    has_attachment: bool = False         # attachments bump the risk score

    class Config:
        extra = "ignore"


class AnalyzeRequest(BaseModel):
    type: str = "email"
    channel: Optional[str] = ""

    # Default to "now" (timezone-aware) if the caller doesn't send a timestamp.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    subject: str = ""

    sender: Optional[Sender] = None
    receiver: Optional[Receiver] = None

    # default_factory gives each request its own fresh empty object/list rather
    # than sharing one mutable default across all requests.
    content: Content = Field(default_factory=Content)

    links: List[Link] = Field(default_factory=list)
    images: List[Image] = Field(default_factory=list)
    attachments: List[Attachment] = Field(default_factory=list)

    metadata: Metadata = Field(default_factory=Metadata)

    class Config:
        extra = "ignore"
