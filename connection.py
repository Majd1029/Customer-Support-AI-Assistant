from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
load_dotenv()

url = (
    f"postgresql+psycopg2://{os.environ['PG_USER']}:{os.environ['PG_PASSWORD']}"
    f"@{os.environ['PG_HOST']}:{os.environ['PG_PORT']}/{os.environ['PG_DB']}"
)

engine = create_engine(url)
with engine.connect() as conn:
    result = conn.execute(text("SELECT version()"))
    print("✅ Connexion réussie :", result.fetchone()[0])