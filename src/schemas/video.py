from pydantic import BaseModel, ConfigDict
import uuid

class VideoBase(BaseModel):
    name: str
    url: str

class VideoResponse(VideoBase):
    id: uuid.UUID  # 1. Change to strictly accept a UUID object
    
    # 2. Tell Pydantic to read from SQLAlchemy class attributes
    model_config = ConfigDict(from_attributes=True) 