class StorySageEntities:
    """
    A class to represent the entities information for a story series.

    Attributes:
        people_by_id (dict): A dictionary mapping people IDs to lists of names.
        people_by_name (dict): A dictionary mapping names to people IDs.
        entity_by_id (dict): A dictionary mapping entity IDs to lists of names.
        entity_by_name (dict): A dictionary mapping names to entity IDs.
    """

    def __init__(self, people_by_id: dict, people_by_name: dict, entity_by_id: dict, entity_by_name: dict):
        """
        Initialize the StorySageEntities with the given data.

        Args:
            people_by_id (dict): A dictionary mapping people IDs to lists of names.
            people_by_name (dict): A dictionary mapping names to people IDs.
            entity_by_id (dict): A dictionary mapping entity IDs to lists of names.
            entity_by_name (dict): A dictionary mapping names to entity IDs.
        """
        self.people_by_id = people_by_id
        self.people_by_name = people_by_name
        self.entity_by_id = entity_by_id
        self.entity_by_name = entity_by_name

# Example usage:
if __name__ == "__main__":
    # Example entities data
    entities_data = {
        "people_by_id": {
            "1_p_0": ["Harry Potter"],
            "1_p_1": ["Hermione Granger"]
        },
        "people_by_name": {
            "Harry Potter": "1_p_0",
            "Hermione Granger": "1_p_1"
        },
        "entity_by_id": {
            "1_e_0": ["Hogwarts"],
            "1_e_1": ["Ministry of Magic"]
        },
        "entity_by_name": {
            "Hogwarts": "1_e_0",
            "Ministry of Magic": "1_e_1"
        }
    }

    # Create an instance of StorySageEntities
    entities = StorySageEntities(
        people_by_id=entities_data["people_by_id"],
        people_by_name=entities_data["people_by_name"],
        entity_by_id=entities_data["entity_by_id"],
        entity_by_name=entities_data["entity_by_name"]
    )

    # Accessing attributes
    print(f"People by ID: {entities.people_by_id}")
    print(f"People by Name: {entities.people_by_name}")
    print(f"Entity by ID: {entities.entity_by_id}")
    print(f"Entity by Name: {entities.entity_by_name}")

    # Example result output:
    """
    People by ID: {'1_p_0': ['Harry Potter'], '1_p_1': ['Hermione Granger']}
    People by Name: {'Harry Potter': '1_p_0', 'Hermione Granger': '1_p_1'}
    Entity by ID: {'1_e_0': ['Hogwarts'], '1_e_1': ['Ministry of Magic']}
    Entity by Name: {'Hogwarts': '1_e_0', 'Ministry of Magic': '1_e_1'}
    """
