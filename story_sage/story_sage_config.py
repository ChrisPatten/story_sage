import yaml

class StorySageConfig:
    """
    A class to represent the configuration information for the Story Sage system.

    Attributes:
        openai_api_key (str): The API key for accessing OpenAI services.
        chroma_path (str): The path for ChromaDB storage.
        chroma_collection (str): The name of the ChromaDB collection.
        entities_path (str): The path to the entities file.
        series_path (str): The path to the series file.
        n_chunks (int): The number of chunks to retrieve.
    """

    def __init__(self, config_path: str):
        """
        Initialize the StorySageConfig with the given configuration file path.

        Args:
            config_path (str): The path to the configuration YAML file.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        required_keys = [
            "OPENAI_API_KEY",
            "CHROMA_PATH",
            "CHROMA_COLLECTION",
            "ENTITIES_PATH",
            "SERIES_PATH",
            "N_CHUNKS"
        ]

        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(f"Missing configuration keys: {', '.join(missing_keys)}")
        
        self.openai_api_key = config["OPENAI_API_KEY"]
        self.chroma_path = config["CHROMA_PATH"]
        self.chroma_collection = config["CHROMA_COLLECTION"]
        self.entities_path = config["ENTITIES_PATH"]
        self.series_path = config["SERIES_PATH"]
        self.n_chunks = config["N_CHUNKS"]