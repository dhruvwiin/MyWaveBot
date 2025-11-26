from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    
    # Core Keys
    PERPLEXITY_API_KEY: str
    ADMIN_API_TOKEN: str
    HASH_SALT: str

    # Database
    DATABASE_URL: str = "sqlite:///./analytics.db"

    # Feature flags and privacy
    STORE_MESSAGE_TEXT: bool = False
    USE_EMBEDDINGS: bool = False

    # University specific
    UNIVERSITY_NAME: str = "Acme University"
    SEARCH_DOMAINS: str = "acme.edu,registrar.acme.edu"

settings = Settings()