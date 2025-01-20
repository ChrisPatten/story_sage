from story_sage import Chunk

results_a = {
    'ids': [
        'series_3|book_11|chapter_0|level_1|chunk_1', 
        'series_3|book_11|chapter_0|level_1|chunk_2', 
    ], 
    'embeddings': None, 
    'documents': ['text', 'text 2'], 
    'uris': None, 
    'data': None, 
    'metadatas': [
        {'book_filename': '11_knife_of_dreams.txt',
         'book_number': 11,
         'chapter_number': 0,
         'children': '',
         'chunk_index': 1,
         'full_chunk': 'text',
         'is_summary': False,
         'level': 1,
         'parents': 'series_3|book_11|chapter_0|level_2|chunk_2',
         'series_id': 3}, 
        {'book_filename': '11_knife_of_dreams.txt',
         'book_number': 11,
         'chapter_number': 0,
         'children': '',
         'chunk_index': 2,
         'full_chunk': 'text 2',
         'is_summary': False,
         'level': 1,
         'parents': 'series_3|book_11|chapter_0|level_2|chunk_2',
         'series_id': 3}, 
    ]
}

results_b = {
    'ids': [[
        'series_3|book_11|chapter_0|level_1|chunk_1', 
        'series_3|book_11|chapter_0|level_1|chunk_2', 
    ]], 
    'embeddings': None, 
    'documents': [['text', 'text 2']], 
    'uris': None, 
    'data': None, 
    'metadatas': [[
        {'book_filename': '11_knife_of_dreams.txt',
         'book_number': 11,
         'chapter_number': 0,
         'children': '',
         'chunk_index': 1,
         'full_chunk': 'text',
         'is_summary': False,
         'level': 1,
         'parents': 'series_3|book_11|chapter_0|level_2|chunk_2',
         'series_id': 3}, 
        {'book_filename': '11_knife_of_dreams.txt',
         'book_number': 11,
         'chapter_number': 0,
         'children': '',
         'chunk_index': 2,
         'full_chunk': 'text 2',
         'is_summary': False,
         'level': 1,
         'parents': 'series_3|book_11|chapter_0|level_2|chunk_2',
         'series_id': 3}, 
    ]]
}

chunks_a = Chunk.from_chroma_results(results_a)
chunks_b = Chunk.from_chroma_results(results_b)

print(chunks_a)
print(chunks_b)