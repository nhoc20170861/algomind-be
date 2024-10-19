import json
import logging
import os
from datetime import datetime, timedelta
from logging import getLogger
from typing import Dict, List

import bcrypt  # type: ignore
import requests
from db_utils import get_db_connection
from dotenv import load_dotenv  # type: ignore
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, EmailStr
from request_models import MermaidRequest, RequestModel, UserRequest
from user_session import ChatSession, ChatSessionManager
from typing import List, Optional
import jwt
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()

# Secret key for JWT
SECRET_KEY = os.getenv("SECRET_KEY", "Android@123")  # Ensure you have a strong secret key

# JWT expiration time (in minutes)
JWT_EXPIRATION_TIME = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="signin")

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)
conn = get_db_connection()
PROXY_PREFIX = os.getenv("PROXY_PREFIX", "/api")
app = FastAPI(root_path=PROXY_PREFIX)

API_KEY = os.getenv("API_KEY")


class ModelKWArgs(BaseModel):
    modelParameter: dict = {
        "temperature": 0.75,
        "max_tokens": 2000,
        "top_p": 0.9,
    }


MODEL = os.getenv("MODEL", "anthropic.claude-3-haiku-20240307-v1:0")

origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# fix the region_name -> us-west-2
session_manager = ChatSessionManager(conn=conn)


MODEL = os.getenv("MODEL", "gemini-1.5-flash")
API_TOKEN = os.environ["API_TOKEN"]

chat_model = ChatGoogleGenerativeAI(
    model=MODEL,
    api_key=API_KEY,
)
# USERS API
class SignInRequest(BaseModel):
    email: str
    password: str

@app.post("/signin")
def sign_in(sign_in_request: SignInRequest):
    email = sign_in_request.email
    password = sign_in_request.password

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT id, email, username, password, birthday, created_at, wallet_name, wallet_address FROM algo_users WHERE email = %s', (email,))
        user_row = cursor.fetchone()

        if not user_row:
            return JSONResponse(status_code=401, content={"statusCode": 401, "body": "Unauthorized"})

        stored_password_hash = user_row[3]
        if not bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            return JSONResponse(status_code=401, content={"statusCode": 401, "body": "Unauthorized"})

        # Create JWT token
        payload = {
            "sub": user_row[0],  # User ID
            "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_TIME)  # Expiration time
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

        # Fetch user's projects
        cursor.execute('SELECT id, user_id, name, description, fund_id, current_fund, deadline, created_at FROM algo_projects WHERE user_id = %s', (user_row[0],))
        projects = cursor.fetchall()

        project_list = [
            {
                "id": project[0],
                "user_id": project[1],
                "name": project[2],
                "description": project[3],
                "fund_id": project[4],
                "current_fund": project[5],
                "deadline": project[6].strftime('%Y-%m-%d %H:%M:%S'),
                "created_at": project[7].strftime('%Y-%m-%d %H:%M:%S')
            }
            for project in projects
        ]

        cursor.execute('SELECT id, name_fund, members, description, created_at FROM algo_funds WHERE user_id = %s', (user_row[0],))
        funds = cursor.fetchall()

        fund_list = [
            {
                "id": fund[0],
                "name_fund": fund[1],
                "members": fund[2],
                "description": fund[3],
                "created_at": fund[4].strftime('%Y-%m-%d %H:%M:%S')
            }
            for fund in funds
        ]

        user_info = {
            "id": user_row[0],
            "email": user_row[1],
            "name": user_row[2],
            "birthday": user_row[4].strftime('%Y-%m-%d'),
            "created_at": user_row[5].strftime('%Y-%m-%d %H:%M:%S'),
            "wallet_name": user_row[6], 
            "wallet_address": user_row[7], 
            "projects": project_list,
            "funds": fund_list,
            "token": token 
        }

        return JSONResponse(status_code=200, content={"statusCode": 200, "body": user_info})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        cursor.close()
        conn.close()

@app.post("/signout")
def sign_out():
    return JSONResponse(status_code=200, content={"statusCode": 200, "body": "Signed out successfully"})



class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str
    birthday: str #YYYY-MM-DD format


@app.post("/register")
def register(register_request: RegisterRequest):
    email = register_request.email
    name = register_request.name
    password = register_request.password
    birthday = register_request.birthday 

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT INTO algo_users (email, username, password, birthday) 
            VALUES (%s, %s, %s, %s) RETURNING id, email, username, birthday, created_at
        ''', (email, name, hashed_password, birthday))
        
        user = cursor.fetchone()
        conn.commit()

        user_info = {
            "id": user[0],
            "email": user[1],
            "name": user[2],
            "birthday": user[3].strftime('%Y-%m-%d'),  
            "created_at": user[4].strftime('%Y-%m-%d %H:%M:%S') 
        }

        return JSONResponse(status_code=200, content={
            "statusCode": 200, 
            "body": "Registered successfully",
            "user": user_info
        })

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Something went wrong")
    
    finally:
        cursor.close()
        conn.close()

class UpdateUserRequest(BaseModel):
    birthday: Optional[datetime] = None
    wallet_name: Optional[str] = None
    wallet_address: Optional[str] = None
    follow_count: Optional[str] = None
    password: Optional[str] = None


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.put("/user/{user_id}")
def update_user(user_id: int, update_request: UpdateUserRequest, current_user: int = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        update_fields = []
        update_values = []

        if user_id != current_user:
            raise HTTPException(status_code=403, detail="Permission denied: You can only update your own profile.")

        if update_request.birthday is not None:
            update_fields.append("birthday = %s")
            update_values.append(update_request.birthday)

        if update_request.wallet_name is not None:
            update_fields.append("wallet_name = %s")
            update_values.append(update_request.wallet_name)

        if update_request.wallet_address is not None:
            update_fields.append("wallet_address = %s")
            update_values.append(update_request.wallet_address)

        if update_request.follow_count is not None:
            update_fields.append("follow_count = %s")
            update_values.append(update_request.follow_count)

        if update_request.password is not None:
            hashed_password = bcrypt.hashpw(update_request.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            update_fields.append("password = %s")
            update_values.append(hashed_password)

        if not update_fields:
            return JSONResponse(status_code=400, content={"statusCode": 400, "message": "No fields provided to update"})

        update_fields.append("updated_at = NOW()")
        update_values.append(user_id)

        update_query = f"UPDATE algo_users SET {', '.join(update_fields)} WHERE id = %s"

        cursor.execute(update_query, tuple(update_values))
        conn.commit()

        return JSONResponse(status_code=200, content={"statusCode": 200, "message": "User information updated successfully"})
    
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        cursor.close()
        conn.close()

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

@app.get("/users", response_model=List[UserResponse])
def get_users(current_user: int = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT id, email, username, birthday, created_at, wallet_name, wallet_address FROM algo_users')
        users = cursor.fetchall()

        user_list = [
            {
                "id": user[0],
                "email": user[1],
                "username": user[2],
                "birthday": user[3].strftime('%Y-%m-%d'),
                "created_at": user[4].strftime('%Y-%m-%d %H:%M:%S'),
                "wallet_name": user[5],
                "wallet_address": user[6]
            }
            for user in users
        ]

        return JSONResponse(status_code=200, content={"statusCode": 200, "body": user_list})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        cursor.close()
        conn.close()

# FUNDS API
class CreateFundRequest(BaseModel):
    name_fund: str
    user_id: int
    members: List[int]
    description: str
    logo: Optional[str] = None  

@app.post("/funds/create")
def create_fund(create_fund_request: CreateFundRequest, user_id: int = Depends(get_current_user)):
    name_fund = create_fund_request.name_fund
    members = create_fund_request.members
    description = create_fund_request.description
    logo = create_fund_request.logo
    created_at = datetime.utcnow()

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT INTO algo_funds (name_fund, user_id, members, description, logo, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (name_fund, user_id, members, description, logo, created_at, created_at))
        conn.commit()

        fund_id = cursor.fetchone()[0]
        return JSONResponse(status_code=200, content={"statusCode": 200, "body": {"fund_id": fund_id}})
    
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        cursor.close()
        conn.close()

@app.put("/funds/update/{fund_id}")
def update_fund(fund_id: int, update_fund_request: CreateFundRequest, user_id: int = Depends(get_current_user)):
    name_fund = update_fund_request.name_fund
    members = update_fund_request.members
    description = update_fund_request.description
    logo = update_fund_request.logo
    updated_at = datetime.utcnow()

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            UPDATE algo_funds
            SET name_fund = %s, members = %s, description = %s, logo = %s, updated_at = %s
            WHERE id = %s
        ''', (name_fund, members, description, logo, updated_at, fund_id))
        conn.commit()

        return JSONResponse(status_code=200, content={"statusCode": 200, "body": "Fund updated successfully"})
    
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        cursor.close()
        conn.close()

@app.get("/funds/user/{user_id}")
def get_funds_by_user(user_id: int, current_user: int = Depends(get_current_user)):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="Access forbidden: you do not have permission to access these funds")

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT id, name_fund, members, description, logo, created_at FROM algo_funds WHERE user_id = %s AND deleted_at IS NULL', (user_id,))
        funds = cursor.fetchall()
        funds_list = [
            {
                "id": fund[0],
                "name_fund": fund[1],
                "members": fund[2],
                "description": fund[3],
                "logo": fund[4],
                "created_at": fund[5].strftime('%Y-%m-%d %H:%M:%S')
            }
            for fund in funds
        ]

        return JSONResponse(status_code=200, content={"statusCode": 200, "body": funds_list})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        cursor.close()
        conn.close()

#PROJECT APIS
class CreateProjectRequest(BaseModel):
    user_id: Optional[int] = None
    name: Optional[str] = None
    description: Optional[str] = None
    fund_id: Optional[int] = None
    current_fund: Optional[int] = None
    fund_raise_total: Optional[int] = None
    fund_raise_count: Optional[int] = None
    deadline: Optional[datetime] = None
    project_hash: Optional[str] = None 
    is_verify: Optional[bool] = None
    status: Optional[str] = None
    linkcardImage: Optional[List[str]] = None
    type: Optional[str] = None  

class ProjectResponse(BaseModel):
    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    fund_id: Optional[int] = None
    current_fund: int
    fund_raise_total: int
    fund_raise_count: int
    deadline: Optional[datetime] = None
    project_hash: str
    is_verify: bool
    status: str
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None
    linkcardImage: Optional[List[str]] = None 
    type: Optional[str] = None  

@app.post("/projects", response_model=ProjectResponse)
def create_project(project_request: CreateProjectRequest, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.now()
    updated_at = created_at

    try:
        cursor.execute('''
            INSERT INTO algo_projects (user_id, name, description, fund_id, current_fund, fund_raise_total, fund_raise_count, 
            deadline, project_hash, is_verify, status, linkcardImage, type, created_at, updated_at) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
        ''', (
            project_request.user_id, 
            project_request.name, 
            project_request.description, 
            project_request.fund_id, 
            project_request.current_fund, 
            project_request.fund_raise_total, 
            project_request.fund_raise_count, 
            project_request.deadline, 
            project_request.project_hash, 
            project_request.is_verify, 
            project_request.status,
            project_request.linkcardImage, 
            project_request.type,  
            created_at, 
            updated_at
        ))

        project_id = cursor.fetchone()[0]
        conn.commit()

        return {
            "id": project_id,
            "user_id": project_request.user_id,
            "name": project_request.name,
            "description": project_request.description,
            "fund_id": project_request.fund_id,
            "current_fund": project_request.current_fund,
            "fund_raise_total": project_request.fund_raise_total,
            "fund_raise_count": project_request.fund_raise_count,
            "deadline": project_request.deadline,
            "project_hash": project_request.project_hash,
            "is_verify": project_request.is_verify,
            "status": project_request.status,
            "linkcardImage": project_request.linkcardImage,  
            "type": project_request.type, 
            "created_at": created_at,
            "updated_at": updated_at
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()



@app.put("/projects/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, project_request: CreateProjectRequest, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    updated_at = datetime.now()

    try:
        cursor.execute('SELECT * FROM algo_projects WHERE id = %s', (project_id,))
        current_project = cursor.fetchone()

        if not current_project:
            raise HTTPException(status_code=404, detail="Project not found.")

        updated_fields = {
            "user_id": project_request.user_id if project_request.user_id is not None else current_project[1],
            "name": project_request.name if project_request.name is not None else current_project[2],
            "description": project_request.description if project_request.description is not None else current_project[3],
            "fund_id": project_request.fund_id if project_request.fund_id is not None else current_project[4],
            "current_fund": project_request.current_fund if project_request.current_fund is not None else current_project[5],
            "fund_raise_total": project_request.fund_raise_total if project_request.fund_raise_total is not None else current_project[6],
            "fund_raise_count": project_request.fund_raise_count if project_request.fund_raise_count is not None else current_project[7],
            "deadline": project_request.deadline if project_request.deadline is not None else current_project[8],
            "project_hash": project_request.project_hash if project_request.project_hash is not None else current_project[9],
            "is_verify": project_request.is_verify if project_request.is_verify is not None else current_project[10],
            "status": project_request.status if project_request.status is not None else current_project[11],
            "linkcardImage": project_request.linkcardImage if project_request.linkcardImage is not None else current_project[12], 
            "type": project_request.type if project_request.type is not None else current_project[13],  
        }

        cursor.execute('''
            UPDATE algo_projects 
            SET user_id = %s, name = %s, description = %s, fund_id = %s, current_fund = %s, 
                fund_raise_total = %s, fund_raise_count = %s, deadline = %s, 
                project_hash = %s, is_verify = %s, status = %s, linkcardImage = %s, type = %s, updated_at = %s 
            WHERE id = %s;
        ''', (
            updated_fields["user_id"],
            updated_fields["name"],
            updated_fields["description"],
            updated_fields["fund_id"],
            updated_fields["current_fund"],
            updated_fields["fund_raise_total"],
            updated_fields["fund_raise_count"],
            updated_fields["deadline"],
            updated_fields["project_hash"],
            updated_fields["is_verify"],
            updated_fields["status"],
            updated_fields["linkcardImage"],
            updated_fields["type"],  
            updated_at,
            project_id
        ))

        conn.commit()

        return {
            "id": project_id,
            "user_id": updated_fields["user_id"],
            "name": updated_fields["name"],
            "description": updated_fields["description"],
            "fund_id": updated_fields["fund_id"],
            "current_fund": updated_fields["current_fund"],
            "fund_raise_total": updated_fields["fund_raise_total"],
            "fund_raise_count": updated_fields["fund_raise_count"],
            "deadline": updated_fields["deadline"],
            "project_hash": updated_fields["project_hash"],
            "is_verify": updated_fields["is_verify"],
            "status": updated_fields["status"],
            "linkcardImage": updated_fields["linkcardImage"],  
            "type": updated_fields["type"],  
            "created_at": current_project[14], 
            "updated_at": updated_at
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()
      

@app.get("/projects/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT * FROM algo_projects WHERE id = %s AND deleted_at IS NULL;', (project_id,))
        project = cursor.fetchone()

        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        return {
            "id": project[0],
            "user_id": project[1],
            "name": project[2],
            "description": project[3],
            "fund_id": project[4],
            "current_fund": project[5],
            "fund_raise_total": project[6],
            "fund_raise_count": project[7],
            "deadline": project[8],
            "project_hash": project[9],
            "is_verify": project[10],
            "status": project[11],
            "linkcardImage": project[15],
            "type": project[16],
            "created_at": project[12],
            "updated_at": project[13],
            "deleted_at": project[14]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/projects", response_model=List[ProjectResponse])
def get_projects(current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT * FROM algo_projects WHERE deleted_at IS NULL ORDER BY id DESC;')
        projects = cursor.fetchall()

        return [{
            "id": project[0],
            "user_id": project[1],
            "name": project[2],
            "description": project[3],
            "fund_id": project[4],
            "current_fund": project[5],
            "fund_raise_total": project[6],
            "fund_raise_count": project[7],
            "deadline": project[8],
            "project_hash": project[9],
            "is_verify": project[10],
            "status": project[11],
            "linkcardImage": project[15], 
            "type": project[16], 
            "created_at": project[12],
            "updated_at": project[13],
            "deleted_at": project[14]
        } for project in projects]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: int, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            UPDATE algo_projects 
            SET deleted_at = %s 
            WHERE id = %s;
        ''', (datetime.now(), project_id))
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Project not found")
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

class UpdateProjectFundingRequest(BaseModel):
    current_fund: float

@app.put("/projects/{project_id}/addFund", response_model=ProjectResponse)
def update_project_funding(project_id: int, funding_request: UpdateProjectFundingRequest, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    updated_at = datetime.now()

    try:
        cursor.execute('SELECT * FROM algo_projects WHERE id = %s', (project_id,))
        current_project = cursor.fetchone()

        if not current_project:
            raise HTTPException(status_code=404, detail="Project not found.")

        current_fund = current_project[5]  
        fund_raise_total = current_project[6] 
        fund_raise_count = current_project[7] 

        new_current_fund = current_fund + funding_request.current_fund
        if new_current_fund > fund_raise_total:
            raise HTTPException(status_code=400, detail="Current fund exceeds the total fundraising goal.")

        cursor.execute('''
            UPDATE algo_projects 
            SET current_fund = %s, updated_at = %s 
            WHERE id = %s;
        ''', (
            new_current_fund,
            updated_at,
            project_id
        ))

        cursor.execute('''
            INSERT INTO algo_contributions (project_id, user_id, amount, created_at, updated_at) 
            VALUES (%s, %s, %s, %s, %s);
        ''', (
            project_id,
            current_user["id"],
            funding_request.current_fund,
            updated_at,
            updated_at
        ))

        conn.commit()

        return {
            "id": project_id,
            "user_id": current_project[1],
            "name": current_project[2],
            "description": current_project[3],
            "fund_id": current_project[4],
            "current_fund": new_current_fund,
            "fund_raise_total": fund_raise_total,
            "fund_raise_count": fund_raise_count + 1,
            "deadline": current_project[8],
            "project_hash": current_project[9],
            "is_verify": current_project[10],
            "status": current_project[11],
            "created_at": current_project[12],
            "updated_at": updated_at
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/health-check")
def health_check():
    return {"status": "ok"}


@app.get("/")
def home():
    return {"message": "Solar Sailors welcome you to the backend of the project."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)