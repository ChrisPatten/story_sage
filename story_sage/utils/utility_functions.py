import logging

def flatten_nested_list(nested_list: list) -> list:
    """Flattens a nested list of arbitrary depth into a single-level list.
    
    This function takes a nested list structure and converts it into a flat list by 
    recursively extracting all non-list elements while preserving their order.
    
    Args:
        nested_list (list): A list that may contain other lists as elements at any depth.
            
    Returns:
        list: A flattened list containing all non-list elements from the input.
        
    Examples:
        >>> nested = [1, [2, 3, [4, 5]], 6]
        >>> flatten_nested_list(nested)
        [1, 2, 3, 4, 5, 6]
        
        >>> nested = [['a', 'b'], [], ['c', ['d', 'e']]]
        >>> flatten_nested_list(nested)
        ['a', 'b', 'c', 'd', 'e']
    """
    def _flatten_generator(nested_list):
        """Helper generator function that recursively yields items from nested lists.
        
        Args:
            nested_list: The nested list structure to flatten.
        
        Yields:
            Individual non-list elements in depth-first order.
        """
        for item in nested_list:
            if isinstance(item, list):
                # Recursively flatten any nested lists
                yield from _flatten_generator(item)
            else:
                # Yield non-list items directly
                yield item

    return list(_flatten_generator(nested_list))

MODULE_EMOJI_MAP = {
    'story_sage.services.chain': '🔗',
    'story_sage.services.retriever': '🐶',
    'story_sage.services.search': '🔎',
    'story_sage.models.chunk': '🧱'
}

def format_log_with_emoji(level_name: str) -> str:
    """Returns appropriate emoji for different log levels"""
    emoji_map = {
        'DEBUG': '🐞',
        'INFO': 'ℹ️',
        'WARNING': '⚠️', 
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    return emoji_map.get(level_name, '📝')

class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emoji to log messages"""
    def format(self, record):
        # Add emoji based on log level
        emoji = format_log_with_emoji(record.levelname)
        record.emoji = emoji
        
        # Add module emoji if available 
        module_emoji = MODULE_EMOJI_MAP.get(record.name, '')
        record.module_emoji = f" {module_emoji}" if module_emoji else ''
        
        return super().format(record)
    
def configure_logging(logger_names: list[str], level: int = logging.INFO):
    """Set up logging for a specific logger"""
    handler = logging.StreamHandler()
    formatter = EmojiFormatter('%(emoji)s %(name)s%(module_emoji)s: %(message)s')
    handler.setFormatter(formatter)

    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        
        # Remove any existing handlers
        for existing_handler in logger.handlers[:]:
            logger.removeHandler(existing_handler)
            
        logger.addHandler(handler)
        print(f"ℹ️ Logging set up for {logger_name} at level {level}")