from app.settings import settings
import tiktoken

tokenizer = tiktoken.encoding_for_model(settings.CHAT_MODEL)


def get_final_score(scores):
    final_score = 0
    if all([s >= 2.5 for s in scores]) and scores[-1] >= 2.75:
        final_score = 3
    elif all([s >= 1.5 for s in scores]) and scores[-1] >= 2:
        final_score = 2
    elif all([s >= 0.5 for s in scores]) and scores[-1] >= 1:
        final_score = 1
    return final_score


def extract_chunks(resource: str):
    tokenized = tokenizer.encode(resource)
    total_chunks = settings.CHUNKS_PER_DOC
    chunk_offset = 0

    # Gather chunks
    if len(tokenized) > total_chunks * settings.TOKENS_PER_CHUNK:
        chunk_offset = (len(tokenized) - (total_chunks * settings.TOKENS_PER_CHUNK)) // (total_chunks + 1)
    else:
        # Lower chunk count to document size
        total_chunks = max(1, len(tokenized) // settings.TOKENS_PER_CHUNK)

    chunks = []
    i = chunk_offset # counter for token position
    for chunk_idx in range(total_chunks):
        # find a smarter split boundary
        start = i
        end = i + settings.TOKENS_PER_CHUNK
        max_dist = settings.MAX_SEARCH_DISTANCE
        start_chunk = ""
        end_chunk = ""

        # Split chunk along newline boundaries if possible
        while start < (i + max_dist) and "\n" not in start_chunk:
            start += 1
            start_chunk = tokenizer.decode(tokenized[start:(start + 2)])

        while end < (i + settings.TOKENS_PER_CHUNK + max_dist) and end < len(tokenized) and "\n" not in end_chunk:
            end += 1
            end_chunk = tokenizer.decode(tokenized[(end - 2):end])

        chunk = tokenizer.decode(tokenized[start:end])
        if "\n" in start_chunk:
            chunk = chunk.split("\n", 1)[1]
        if "\n" in end_chunk:
            chunk = chunk.rsplit("\n", 1)[0]

        chunks.append(chunk)
        i += chunk_offset + settings.TOKENS_PER_CHUNK
    chunks = [c.strip() for c in chunks if len(c.strip()) > settings.TOKENS_PER_CHUNK // 4]
    return chunks
