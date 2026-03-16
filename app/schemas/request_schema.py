from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime


class Sender(BaseModel):
    name: Optional[str] = ""
    email: Optional[EmailStr] = None
    domain: Optional[str] = ""

    class Config:
        extra = "ignore"


class Receiver(BaseModel):
    email: Optional[EmailStr] = None

    class Config:
        extra = "ignore"


class Content(BaseModel):
    text: str = Field(default="", min_length=0)
    html: Optional[str] = ""

    class Config:
        extra = "ignore"


class Link(BaseModel):
    url: str
    domain: Optional[str] = ""
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
    has_attachment: bool = False

    class Config:
        extra = "ignore"


class AnalyzeRequest(BaseModel):

    type: str = "email"
    channel: Optional[str] = ""

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    subject: str = ""

    sender: Optional[Sender] = None
    receiver: Optional[Receiver] = None

    content: Content = Field(default_factory=Content)

    links: List[Link] = Field(default_factory=list)
    images: List[Image] = Field(default_factory=list)
    attachments: List[Attachment] = Field(default_factory=list)

    metadata: Metadata = Field(default_factory=Metadata)

    class Config:
        extra = "ignore"