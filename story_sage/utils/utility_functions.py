

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