from sqlmodel import Field, SQLModel, Relationship
from sqlalchemy import Column, Integer, String, DateTime, ARRAY, types
from sqlalchemy.sql import func
from typing import Optional, List
from datetime import datetime

# Bảng AlgoUser
class AlgoUser(SQLModel, table=True):
    __tablename__ = "algo_users"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    email: str
    birthday: Optional[datetime]
    password: str  # Hash mật khẩu
    follow_count: Optional[str]
    created_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), nullable=True
        )
    )
    updated_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), nullable=True
        )
    )
    deleted_at: Optional[datetime]  # Không sử dụng nhưng vẫn khai báo


# Bảng AlgoFund
class AlgoFund(SQLModel, table=True):
    __tablename__ = "algo_funds"
    id: Optional[int] = Field(default=None, primary_key=True)
    name_fund: str
    user_id: int = Field(foreign_key="algo_users.id")
    members: List[int] = Field(default=[], sa_column=Column(ARRAY(Integer)))
    logo: Optional[str]
    description: Optional[str]
    created_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), nullable=True
        )
    )
    updated_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), nullable=True
        )
    )
    deleted_at: Optional[datetime]  # Không sử dụng nhưng vẫn khai báo

    user: Optional[AlgoUser] = Relationship()
    
    class Config:
        arbitrary_types_allowed = True  # Cho phép loại tùy ý
    


# Bảng AlgoProject
class AlgoProject(SQLModel, table=True):
    __tablename__ = "algo_projects"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="algo_users.id")
    name: str
    description: str  # Có thể chứa Markdown
    fund_id: int = Field(foreign_key="algo_funds.id")
    current_fund: Optional[int]
    linkcardImage: Optional[List[str]] = Field(default=[], sa_column=Column(ARRAY(String)))
    type: Optional[str]
    fund_raise_total: Optional[int]
    fund_raise_count: Optional[int]
    deadline: Optional[datetime]
    wallet_address: str  # Algorand wallet address
    wallet_type: Optional[str]  # Pera, AlgoSigner
    is_verify: bool = False
    status: Optional[str]
    created_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), nullable=True
        )
    )
    updated_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), nullable=True
        )
    )
    deleted_at: Optional[datetime]  # Không sử dụng nhưng vẫn khai báo

    user: Optional[AlgoUser] = Relationship()
    fund: Optional[AlgoFund] = Relationship()

    class Config:
        arbitrary_types_allowed = True  # Cho phép loại tùy ý
    


# Bảng AlgoProjectTrack
class AlgoProjectTrack(SQLModel, table=True):
    __tablename__ = 'algo_project_tracks'
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="algo_projects.id")
    user_id: Optional[int] = Field(default=None, foreign_key="algo_users.id")
    created_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), nullable=True
        )
    )
    updated_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), nullable=True
        )
    )
    deleted_at: Optional[datetime]  # Không sử dụng nhưng vẫn khai báo

    project: Optional[AlgoProject] = Relationship()
    user: Optional[AlgoUser] = Relationship()


# Bảng AlgoContributionTransaction
class AlgoContributionTransaction(SQLModel, table=True):
    __tablename__ = 'algo_contributions_transtaction'
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="algo_projects.id")
    txid: str
    amount: float
    email: Optional[str]
    sodienthoai: Optional[str]
    address: Optional[str]
    name: Optional[str]
    type_sender_wallet: Optional[str]
    sender_wallet_address: Optional[str]
    project_wallet_address: Optional[str] #= Field(default=None, foreign_key="algo_projects.wallet_address")  # Tham chiếu đến wallet_address của AlgoProject
    time_round: Optional[datetime]
    created_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), nullable=True
        )
    )
    updated_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), nullable=True
        )
    )

    project: Optional[AlgoProject] = Relationship()


# Bảng AlgoReceiver
class AlgoReceiver(SQLModel, table=True):
    __tablename__ = 'algo_receivers'
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="algo_projects.id")
    email: Optional[str]
    sodienthoai: Optional[str]
    address: Optional[str]
    name: Optional[str]
    type_receiver_wallet: Optional[str]
    receiver_wallet_address: Optional[str]
    created_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), nullable=True
        )
    )
    updated_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), nullable=True
        )
    )

    project: Optional[AlgoProject] = Relationship()


# Bảng AlgoReceiverTransaction
class AlgoReceiverTransaction(SQLModel, table=True):
    __tablename__ = 'algo_receivers_transtaction'
    id: Optional[int] = Field(default=None, primary_key=True)
    receiver_id: int = Field(foreign_key="algo_receivers.id")
    project_id: int = Field(foreign_key="algo_projects.id")
    transaction_count: Optional[int]
    amount: float
    time_round: int
    created_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), nullable=True
        )
    )
    updated_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), nullable=True
        )
    )

    receiver: Optional[AlgoReceiver] = Relationship()
    project: Optional[AlgoProject] = Relationship()
