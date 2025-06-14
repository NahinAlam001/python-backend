from pydantic import BaseModel
from pydantic.networks import EmailStr

class UserSchema(BaseModel):
    email: EmailStr
    password: str

class UserInfoSchema(BaseModel):
    email: EmailStr

    class Config:
        orm_mode = True

class UserData(BaseModel):
    id:int

class Post(BaseModel):
    title:str
    content:str

    class Config:
        orm_mode = True
