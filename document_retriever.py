import attr

from typing import Any, List, Union, Tuple
import json
import attr
import numpy as np
from sentence_transformers import SentenceTransformer
import scann
import constants
import os

@attr.s
class RetrievalArgs:
    trained_vector_db_file_path: str = attr.ib(default=constants.TRAINED_VECTOR_DB_PATH)
    vector_db_index_to_papers_map_file_path: str = attr.ib(default=constants.EMBEDDING_INDEX_TO_PAPER_FILE_PATH)
    paper_text_files_path: str = attr.ib(default=constants.CLEANED_TEXT_FOLDER_PATH)
    top_k: int = attr.ib(default=5)


class DocumentRetriever:
    def __init__(self, retrieval_args: RetrievalArgs = None, query_embedding_model: "SentenceTransformer" = None):
        ## TODO: We would have to ensure that we have a wrapper over vector DB to have the same API exposed irrespective
        # of the underlying DB so the find_similar_paper doesn't need to know the underlying model and crresponding API
        # to search for nearest neighbors.
        # This will allow us to pass a trained and instantiated vector_db directly as an argument without having to
        # load it from the path.
        if retrieval_args is None:
            retrieval_args = RetrievalArgs()

        self.vector_db = self.load_pre_trained_vector_db(retrieval_args.trained_vector_db_file_path)
        self.paper_text_files_path = retrieval_args.paper_text_files_path
        self.top_k = retrieval_args.top_k

        if query_embedding_model is None:
            query_embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.model = query_embedding_model

        self.load_idx_to_paper_maps(retrieval_args.vector_db_index_to_papers_map_file_path)

    @staticmethod
    def load_pre_trained_vector_db(path: str):
        ''' It currently assumes the SCANN Model from Google and relies on it's load API. However, this can be generalized once there are wrapper around these different model classes.
        '''
        trained_vector_index = scann.scann_ops_pybind.load_searcher(path)
        return trained_vector_index

    def load_idx_to_paper_maps(self, file_name: str = constants.EMBEDDING_INDEX_TO_PAPER_FILE_PATH):
        '''Loads the pre-saved mapping between the index of the emebdding as stored in the SCANN vector DB to the corresponding paper name and the file containing the actual abstract and the body text of the paper.
        '''
        self.vector_index_to_paper_map = {}
        with open(file_name, 'r') as f:
            self.vector_index_to_paper_map = json.loads(f.readlines()[0])

    def parse_query_to_embedding(self, query: str) -> np.array:
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        return query_embedding

    def find_similar_papers(self, query_embedding: str, top_k: int = None) -> Tuple[Any, Any]:
        '''Finds the top_k similar embeddings and returns a sorted List of tuples of their index to the similarity score in descending order of scores.
        '''
        if top_k is None:
            top_k = self.top_k

        indexes, scores = self.vector_db.search(query_embedding, final_num_neighbors=top_k)
        return indexes, scores

    def retrieve_candidate_papers_for_query(self, query: Union[str, np.array], k: int = None) -> List[Tuple[str, str]]:
        '''Uses the trained SCANN database to retrive the top K neighbors based on similarity with the embeddings for the title + abstract section of the paper.
          Fetches paper
        '''
        if isinstance(query, np.ndarray):
            query_emebedding = query
        elif isinstance(query, str):
            query_embedding = self.parse_query_to_embedding(query)

        candidate_indexes, candidate_score = self.find_similar_papers(query_embedding, k)
        if len(candidate_indexes) == 0:
            return []

        candidate_papers = []
        for idx, score in zip(candidate_indexes, candidate_score):
            paper_file_name = self.vector_index_to_paper_map.get(str(idx), None)
            if paper_file_name:
                candidate_papers.append(paper_file_name)
        return candidate_papers

    # These 2 functions should be Retrieval class and hence Retrieval also has to be attached to the Agent class.
    # The reasons this should be in retrieval so it can be independently access of the agent. As well as this is closely
    # tied to the instantiation of the retrieval based on the index to file map file used, vector database used etc.
    # so the retrieval and getting the text has to linked. Also there are scenarios where we wouldn't needs these since
    # the vector database would automatically do this under the hood.
    def get_abstract_from_paper_file_name(self, paper_file_name: Tuple[str, str]):
        paper_name = paper_file_name[0]
        file_name = paper_file_name[1]
        file_path = self.paper_text_files_path + file_name
        with open(file_path, 'r') as f:
            papers = json.loads(f.readlines()[0])
            paper_text = papers.get(paper_name, {})

        return paper_text.get('abstract', '')

    def get_body_from_paper_file_name(self, paper_file_name: Tuple[str, str]):
        paper_name = paper_file_name[0]
        file_name = paper_file_name[1]
        file_path = self.paper_text_files_path + file_name
        with open(file_path, 'r') as f:
            papers = json.loads(f.readlines()[0])
            paper_text = papers.get(paper_name, {})

        return paper_text.get('main_body', '')
