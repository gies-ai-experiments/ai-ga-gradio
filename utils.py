import os
import shutil
import time

from langchain import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from custom_csv_loader import CSVLoader


def reset_folder(destination):
    # synchrnously and recursively delete the destination folder and all its contents, donot return until done
    if os.path.isdir(destination):
        shutil.rmtree(destination)
        while os.path.isdir(destination):
            time.sleep(4)
    os.mkdir(destination)
    while not os.path.isdir(destination):
        time.sleep(4)


def search_index_from_docs(source_chunks, embeddings):
    # print("source chunks: " + str(len(source_chunks)))
    # print("embeddings: " + str(embeddings))
    search_index = FAISS.from_documents(source_chunks, embeddings)
    return search_index


def load_index(folder_path, index_name, embeddings):
    # Load index
    db = FAISS.load_local(
        folder_path=folder_path,
        index_name=index_name, embeddings=embeddings,
    )
    print("Loaded index")
    return db


def fetch_data_for_embeddings(document_list):
    print("document list: " + str(len(document_list)))
    return document_list


def create_chunk_documents(document_list):
    sources = fetch_data_for_embeddings(document_list)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    source_chunks = splitter.split_documents(sources)

    print("chunks: " + str(len(source_chunks)))
    print("sources: " + str(len(sources)))

    return source_chunks


def create_index(folder_path, index_name, embeddings, document_list):
    source_chunks = create_chunk_documents(document_list)
    search_index = search_index_from_docs(source_chunks, embeddings)
    FAISS.save_local(search_index, folder_path=folder_path, index_name=index_name)
    return search_index


def get_csv_files(csv_file, source_column, field_names=None):
    loader = None
    if field_names:
        loader = CSVLoader(file_path=csv_file, source_column=source_column,
                           csv_args={'fieldnames': field_names, 'restkey': 'restkey'})
    else:
        loader = CSVLoader(file_path=csv_file, source_column=source_column, )
    document_list = loader.load()
    return document_list


def index_exists(pickle_file, index_file):
    return os.path.isfile(pickle_file) and os.path.isfile(index_file) and os.path.getsize(
        pickle_file) > 0
