from fastapi import APIRouter, Depends, HTTPException, status, Depends
from typing import List
from app.utils import hash_password
from .. import models, schemas, oauth2
from sqlalchemy.orm import Session
from ..database import get_db


router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.get('/', response_model=List[schemas.UserInfoSchema])
async def root(db: Session = Depends(get_db)):
    user = db.query(models.User).all()
    return user

@router.post('/')
def create_user(user:schemas.UserSchema, db: Session = Depends(get_db), logged_user=Depends(oauth2.get_current_user_id)):
    new_user_dict = user.dict()
    new_user_dict['password'] = hash_password(user.password)
    new_user = models.User(**new_user_dict)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.get('/{id}')
def get_user(id:int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User with id {id} not found")
    return user

@router.put('/{id}')
def update_user(id:int, user:schemas.UserSchema, db: Session = Depends(get_db)):
    user_to_update = db.query(models.User).filter(models.User.id == id).first()
    if not user_to_update:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User with id {id} not found")
    user_to_update.email = user.email
    user_to_update.password = user.password
    db.commit()
    db.refresh(user_to_update)
    return user_to_update

@router.delete('/{id}')
def delete_user(id:int, db: Session = Depends(get_db)):
    user_to_delete = db.query(models.User).filter(models.User.id == id).first()
    if not user_to_delete:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User with id {id} not found")
    db.delete(user_to_delete)
    db.commit()
    return {"message": f"User with id {id} deleted"}
