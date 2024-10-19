import json
import logging
import os
from datetime import datetime
from logging import getLogger
from typing import Dict, List

import bcrypt  # type: ignore
import requests
from db_utils import get_db_connection
from dotenv import load_dotenv  # type: ignore
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, EmailStr
from request_models import MermaidRequest, RequestModel, UserRequest
from user_session import ChatSession, ChatSessionManager

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


class RequestModel(ModelKWArgs):
    userID: str
    requestID: str
    user_input: str


class MermaidRequest(BaseModel):
    userID: str
    requestID: str


class RequestModelProject(ModelKWArgs):
    id: str
    text: str
    response: str


class UserRequest(ModelKWArgs):
    userID: str
    requestID: str
    user_input: List[RequestModelProject]


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
        cursor.execute('SELECT id, email, username, password, birthday, created_at FROM algo_users WHERE email = %s', (email,))
        user_row = cursor.fetchone()

        if not user_row:
            return JSONResponse(status_code=401, content={"statusCode": 401, "body": "Unauthorized"})

        stored_password_hash = user_row[3] 
        if not bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            return JSONResponse(status_code=401, content={"statusCode": 401, "body": "Unauthorized"})

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
        user_info = {
            "id": user_row[0],
            "email": user_row[1],
            "name": user_row[2],
            "birthday": user_row[4].strftime('%Y-%m-%d'),
            "created_at": user_row[5].strftime('%Y-%m-%d %H:%M:%S'),
            "projects": project_list
        }

        return JSONResponse(status_code=200, content={
            "statusCode": 200,
            "body": user_info
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()


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

# Project API
class Prompt(BaseModel):
    id: str
    text: str
    response: str

class AddProjectRequest(BaseModel):
    userId: int
    title: str
    prompts: List[Prompt]
    mermaid: str

@app.post("/api/projects/addProject")
def add_project(add_project_request: AddProjectRequest):
    user_id = add_project_request.userId
    title = add_project_request.title
    prompts = add_project_request.prompts
    mermaid = add_project_request.mermaid

    prompts_json = json.dumps([prompt.dict() for prompt in prompts])

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT INTO "project" (userId, title, prompts, mermaid) 
            VALUES (%s, %s, %s::json, %s) RETURNING id
        ''', (user_id, title, prompts_json, mermaid))

        project_id = cursor.fetchone()[0]

        conn.commit()

        return JSONResponse(
            status_code=200,
            content={
                "statusCode": 200,
                "body": {
                    "project_id": project_id
                }
            }
        )

    except Exception as e:
        logger.error(f"Error adding project: {str(e)}")  # Log the error
        conn.rollback()
        return JSONResponse(status_code=500, content={"statusCode": 500, "body": "Something Went Wrong"})
    finally:
        cursor.close()
        conn.close()



class DeleteProjectRequest(BaseModel):
    projectId: int
    userId: int

@app.delete("/api/projects/deleteProject")
def delete_project(delete_project_request: DeleteProjectRequest):
    project_id = delete_project_request.projectId
    user_id = delete_project_request.userId

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            DELETE FROM "project" WHERE id = %s AND userId = %s
        ''', (project_id, user_id))

        if cursor.rowcount == 0:
            return JSONResponse(status_code=404, content={"statusCode": 404, "body": "Project not found or unauthorized"})

        conn.commit()

        return JSONResponse(status_code=200, content={"statusCode": 200, "body": "Project Deleted"})

    except Exception as e:
        logger.error(f"Error deleting project: {str(e)}") 
        conn.rollback()
        return JSONResponse(status_code=500, content={"statusCode": 500, "body": "Something Went Wrong"})
    finally:
        cursor.close()
        conn.close()

class GetAllProjectsRequest(BaseModel):
    userId: int

@app.post("/api/projects/getAllProjects")
def get_all_projects(get_projects_request: GetAllProjectsRequest):
    user_id = get_projects_request.userId

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT * FROM "project" WHERE "userid" = %s
        ''', (user_id,))

        projects = cursor.fetchall()

        project_list = []
        for project in projects:
            project_dict = {
                "id": project[0],
                "userId": project[1],
                "title": project[2],
                "prompts": project[3],
                "mermaid": project[4]
            }
            project_list.append(project_dict)

        return JSONResponse(status_code=200, content={"statusCode": 200, "body": {"projects": project_list}})

    except Exception as e:
        logger.error(f"Error retrieving projects: {str(e)}")
        return JSONResponse(status_code=500, content={"statusCode": 500, "body": "Something Went Wrong"})
    finally:
        cursor.close()
        conn.close()

class GetOneProjectRequest(BaseModel):
    userId: int
    projectId: int


@app.post("/api/projects")
def get_all_projects(get_projects_request: GetOneProjectRequest):
    user_id = get_projects_request.userId
    project_id = get_projects_request.projectId

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT * FROM "project" WHERE "userid" = %s
        ''', (user_id,))

        projects = cursor.fetchall()

        project_list = []
        for project in projects:
            project_dict = {
                "id": project[0],
                "userId": project[1],
                "title": project[2],
                "prompts": project[3],
                "mermaid": project[4]
            }
            if project_id == project[0]:
                project_list.append(project_dict)

        return JSONResponse(status_code=200, content={"statusCode": 200, "body": {"projects": project_list}})

    except Exception as e:
        logger.error(f"Error retrieving projects: {str(e)}")
        return JSONResponse(status_code=500, content={"statusCode": 500, "body": "Something Went Wrong"})
    finally:
        cursor.close()
        conn.close()

class RenameProjectRequest(BaseModel):
    projectId: int
    userId: int
    newTitle: str

@app.put("/api/projects/renameProject")
def rename_project(rename_project_request: RenameProjectRequest):
    project_id = rename_project_request.projectId
    user_id = rename_project_request.userId
    new_title = rename_project_request.newTitle

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            UPDATE "project" SET title = %s WHERE id = %s AND userId = %s
        ''', (new_title, project_id, user_id))

        if cursor.rowcount == 0:
            return JSONResponse(status_code=404, content={"statusCode": 404, "body": "Project not found or unauthorized"})

        conn.commit()

        return JSONResponse(status_code=200, content={"statusCode": 200, "body": "Project Renamed"})

    except Exception as e:
        logger.error(f"Error renaming project: {str(e)}")
        conn.rollback()
        return JSONResponse(status_code=500, content={"statusCode": 500, "body": "Something Went Wrong"})
    finally:
        cursor.close()
        conn.close()

class SaveMermaidRequest(BaseModel):
    projectId: int
    mermaid: str

@app.get("/health-check")
def health_check():
    return {"status": "ok"}


@app.get("/")
def home():
    return {"message": "Solar Sailors welcome you to the backend of the project."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)