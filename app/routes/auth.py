from fastapi.security import OAuth2PasswordRequestForm
from fastapi import APIRouter, Depends, HTTPException
from app import models, oauth2, utils
from app.database import get_db
from sqlalchemy.orm import Session

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

@router.post("/login")
def login(user_login: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == user_login.username).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not utils.verify_password(user_login.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = oauth2.create_access_token(data={"data": user.id})
    return {"access_token": access_token, "token_type": "bearer"}
