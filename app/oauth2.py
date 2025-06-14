from jose import jwt, JWTError
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException
from app.database import get_db
from app.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/auth/login')

SECRET_KEY = "sdasdhjcnasdhqw213c,cdawdasdlaksdl"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt

def verify_access_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        id: str = payload.get("data")

        if id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    return id

def get_current_user_id(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    id = verify_access_token(token, credentials_exception=HTTPException(status_code=401, detail="Invalid token"))
    user = db.query(User).filter(User.id == id).first()
    return user
