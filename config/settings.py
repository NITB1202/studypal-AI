from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    model: str
    model_max_tokens: int
    embedding_model: str
    vector_store_path: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
