from typing import Any
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from octoai.client import OctoAI
import logging
from env_config import read_env_config, set_env
from os import environ
from langchain_community.embeddings import OctoAIEmbeddings
import requests, json
from typing import List

logger = logging.getLogger("client_utils")

load_dotenv()  # take environment variables from .env.

class CustomOctoAIEmbeddings(OctoAIEmbeddings):
    def _compute_embeddings(
        self, texts: List[str], instruction: str
    ) -> List[List[float]]:
        """Compute embeddings using an OctoAI instruct model."""
        embedding = []
        embeddings = []

        for text in texts:
            try:
                resp = requests.post(self.endpoint_url, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.octoai_api_token}"}, data=json.dumps({"input": str([text]), "model": "thenlper/gte-large"}))
                resp_json = resp.json()
                if "embeddings" in resp_json:
                    embedding = resp_json["embeddings"]
                elif "data" in resp_json:
                    json_data = resp_json["data"]
                    for item in json_data:
                        if "embedding" in item:
                            embedding = item["embedding"]

            except Exception as e:
                raise ValueError(f"Error raised by the inference endpoint: {e}") from e

            embeddings.append(embedding)

        return embeddings


def build_octoai_client(**kwargs: Any) -> OctoAI:
    """
    Build OctoAI client based on the environment variables.
    """

    env = read_env_config("COMPLETION")
    with set_env(**env):
        if is_azure():
            client = AzureOpenAI(**kwargs)
        else:
            client = OctoAI(**kwargs)
        return client

def build_langchain_embeddings(**kwargs: Any) -> OctoAIEmbeddings:
    """
    Build OpenAI embeddings client based on the environment variables.
    """

    env = read_env_config("EMBEDDING")

    with set_env(**env):
        if is_azure():
            client = AzureOpenAIEmbeddings(**kwargs)
        else:
            client = CustomOctoAIEmbeddings(**kwargs)
        return client

def is_azure():
    azure = "AZURE_OPENAI_ENDPOINT" in environ or "AZURE_OPENAI_KEY" in environ or "AZURE_OPENAI_AD_TOKEN" in environ
    if azure:
        logger.debug("Using Azure OpenAI environment variables")
    else:
        logger.debug("Using OctoAI environment variables")
    return azure
