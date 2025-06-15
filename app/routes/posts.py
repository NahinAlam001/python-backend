from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import models, schemas, oauth2
from ..database import get_db

router = APIRouter(
    prefix="/posts",
    tags=["Posts"]
)

@router.get('/', response_model=schemas.UserInfoSchema)
def view_posts(db: Session = Depends(get_db)):
    users = db.query(models.Posts).all()
    return users

@router.post('/')
def create_post(post:schemas.Post, db:Session = Depends(get_db), logged_user =  Depends(oauth2.get_current_user_id)):
    post_dict = post.dict()
    post_dict['user_id'] = logged_user.id
    db.add(models.Posts(**post_dict))
    db.commit()
    return post
