CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    source text NOT NULL,
    chunk_number integer NOT NULL,
    content text NOT NULL,
    embedding vector(1536) NOT NULL,
    PRIMARY KEY (source, chunk_number)
);
