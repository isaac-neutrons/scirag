"""Configuration for RavenDB connection."""

import os

from dotenv import load_dotenv

from scirag.constants import DEFAULT_RAVENDB_DATABASE, DEFAULT_RAVENDB_URL

# Load environment variables
load_dotenv()


class RavenDBConfig:
    """Configuration class for RavenDB connection details."""

    @staticmethod
    def get_url() -> str:
        """Get the RavenDB server URL from environment variables.

        Returns:
            str: RavenDB server URL (default: http://localhost:8080)
        """
        return os.getenv("RAVENDB_URL", DEFAULT_RAVENDB_URL)

    @staticmethod
    def get_database_name() -> str:
        """Get the RavenDB database name from environment variables.

        Returns:
            str: Database name (default: scirag)
        """
        return os.getenv("RAVENDB_DATABASE", DEFAULT_RAVENDB_DATABASE)
