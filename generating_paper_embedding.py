import numpy as np
import os
from typing import List
import json
from sentence_transformers import SentenceTransformer

import constants
import annoy

CHUNK_SIZE = 100
EMPTY_EMBEDDING = np.zeros(384)


def split_into_chunks(txt: List[str]) -> List[str]:
    if len(txt) < CHUNK_SIZE:
        return [' '.join(word for word in txt)]

    txt_len = len(txt)
    chunked_txt = []
    for idx in range(0, txt_len, CHUNK_SIZE):
        chunked_txt.append(' '.join(txt[idx:idx + CHUNK_SIZE]))
    return chunked_txt


def get_abstract_embeddings(abstract, model) -> np.array:
    '''Since the embedding generation model only accepts 384 tokens, to avoid loosing critical information from abstract
       we split the abstract into chunks of 100 so we can capture maximum information with minimal truncation.
       Mean of all the chunks are used as a representation of the abstract.
    '''
    split_abstract = split_into_chunks(abstract)
    # Encode the sentences and get a tensor of embeddings
    embeddings = model.encode(split_abstract, normalize_embeddings=True)
    # Calculate the mean embedding
    mean_embedding = np.mean(embeddings, dim=0)
    return mean_embedding


def get_body_embeddings(body_text, model) -> np.array:
    ''' Since most lines in the body text have <100 word tokens it was considered safe to generate embedding for every
    line and represent the body as a 2-d array of num_lines * embedding_dimension
    '''
    if body_text == '' or body_text is None:
        return EMPTY_EMBEDDING
    body_txt_lines = body_text.splitlines()
    # Encode the sentences and get a tensor of embeddings
    embeddings = model.encode(body_txt_lines, normalize_embeddings=True)
    return embeddings


def generate_embedding_from_text(model):
    overall_idx = 0
    abstract_embeddings_collection = {}
    body_embeddings_collection = {}
    for idx, file_name in enumerate(os.listdir(constants.CLEANED_TEXT_FOLDER_PATH)):
        with open(file=os.path.join(constants.CLEANED_TEXT_FOLDER_PATH,file_name), mode='r') as f:
            content = json.loads(f.readlines()[0])
        print(f"Finished reading text from file : {file_name}")
        base_file = file_name.split('_')[-1].split('.')[0]
        print(f"base file index is. : {base_file}")
        for name, paper in content.items():
            overall_idx += 1
            abstract = paper['abstract']
            body_text = paper['main_body']
            body_embed = get_body_embeddings(body_text, model)
            abstract_embed = get_abstract_embeddings(abstract, model)
            abstract_embeddings_collection[name] = abstract_embed.tolist()
            body_embeddings_collection[name] = body_embed.tolist()
            if overall_idx != 0 and overall_idx % 100 == 0:
                print(F'Finished computing embeddings for : {overall_idx} papers')
            if overall_idx != 0 and overall_idx % 1000 == 0:
                split_name = int(overall_idx // 1000)

                print(f'Finish computing embeddings for {base_file, overall_idx} files')
                abstract_embeddings_file_name = constants.ABSTRACT_EMBEDDINGS_FILE_PATH.format(base_file, split_name)
                body_embeddings_file_name = constants.BODY_EMBEDDINGS_FILE_PATH.format(base_file, split_name)
                with open(abstract_embeddings_file_name, 'x') as f_abstract:
                    f_abstract.write(json.dumps(abstract_embeddings_collection))
                    print(f"Finished writing abstract file {abstract_embeddings_file_name}")

                with open(body_embeddings_file_name, 'x') as f_body:
                    f_body.write(json.dumps(body_embeddings_collection))
                    print(f"Finished writing body file {body_embeddings_file_name}")

                abstract_embeddings_collection = {}
                body_embeddings_collection = {}
                print('-*' * 40)


if __name__ == "__main__":

    model = SentenceTransformer('all-mpnet-base-v2')
    generate_embedding_from_text(model)
