
import os
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from pypdf.errors import PdfStreamError
from random import randint


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--reset", action="store_true", help="Reset the database.")
    # args = parser.parse_args()
    # if args.reset:
    #     print("✨ Clearing Database")
    # clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    pdfs = load_pdf()
    # table = load_table()
    chunks = split_documents(documents)
    pdf_chunks = split_documents(pdfs)
    # table_chunks = split_documents(table)
    add_to_chroma(chunks)
    add_to_chroma(pdf_chunks)
    # add_to_chroma(table_chunks)

def add_tuple_to_chroma(tuple):
    timestamp = tuple['timestamp']
    date, time = timestamp.split(' ')
    
    # Extract the parameters
    key = tuple['key']
    N = round(tuple['N'], 3)
    P = round(tuple['P'], 3)
    K = round(tuple['K'], 3)
    humidity = round(tuple['humidity'], 3)
    temperature = round(tuple['temperature'], 3)
    soil_moisture = round(tuple["soilMoisture"], 3)
    crop_name = tuple['crop_name']
    formatted_string = ""
    
    # Create the formatted string
    if (N == 0 and P == 0 and K == 0):
        formatted_string = (
        f"For my {crop_name}, "
        f"at time {time} on {date} the soil moisture was {soil_moisture}, "
        f"humidity was {humidity}, and temperature was {temperature}.\n"
    )
    else:
        formatted_string = (
            f"For my {crop_name} crop, "
            f"at time {time} on {date} the Nitrogen value (n value) was {N}, "
            f"Phosphorus value (p value) was {P}, potassium value (k value) was {K}, soil moisture was {soil_moisture}, "
            f"humidity was {humidity}, and temperature was {temperature}.\n"
        )

    file_name = f"data/temp{key * randint(1, 14)}.txt"
    with open(file_name, "w") as f:
        f.write(formatted_string)
    documents = TextLoader(file_name).load()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    os.remove(file_name)

def load_table():
    table_loader = TextLoader("data/sensor_log.txt")
    return table_loader.load()

def load_documents():
    path = 'data'
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()


def load_pdf():
    pdf_Loader = PyPDFDirectoryLoader("data")
    return pdf_Loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        try:
            db.add_documents(new_chunks, ids=new_chunk_ids)
        except PdfStreamError as e:  # Catching PdfStreamError specifically
            print(f"🚨 Error adding documents to the database: {e}")
        # Optionally, log the ID of the chunk or file causing the error
            print(f"Skipping file due to error: {new_chunk_ids}")
        except Exception as e:  # Catching any other exceptions
            print(f"🚨 An unexpected error occurred: {e}")
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
