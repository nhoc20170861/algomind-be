# initialize_db.py
from sqlmodel import SQLModel, create_engine
from database.models import *  # Giả sử models.py chứa các model của bạn

# Khởi tạo engine PostgreSQL
engine = create_engine("postgresql://anhquan:password@localhost:5432/algomind")

# Tạo lại các bảng trong cơ sở dữ liệu
SQLModel.metadata.create_all(engine, checkfirst=False)


#alembic revision --autogenerate -m "Initial migration"
# 
# alembic upgrade head   


# DO $$ DECLARE
#     r RECORD;
# BEGIN
#     FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
#         EXECUTE 'DROP TABLE ' || quote_ident(r.tablename) || ' CASCADE';
#     END LOOP;
# END $$;DO $$ DECLARE
#     r RECORD;
# BEGIN
#     FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
#         EXECUTE 'DROP TABLE ' || quote_ident(r.tablename) || ' CASCADE';
#     END LOOP;
# END $$;

