import os

SYSTEM_PROMPT_TEMPLATE = """
You are a medical researcher tasked with providing the best answer for the question from the provided medical research paper:
--- START PAPER ---
{paper_text}
--- END PAPER ---
[SEP]
## Instructions
You should provide accurate and informative responses to any question a human may ask about this paper. If you cannot find the relevant answer in the given paper, return saying CANNOT FIND THE ANSWER IN THE PROVIDED SET OF PAPERS.
DO NOT GENERATE ANSWERS BEYOND THE SCOPE OF THE PROVIDED PAPER. RESTRICT THE ANSWER TO A MAXIMUM OF 1000 TOKENS.
"""

HUMAN_QUESTION = "{question}"

ACTOR_USER = "user"
SYSTEM_ROLE = "system"
ASSISTANT_ROLE = "assistant"

if os.getenv('BASE_DIR', ''):
    BASE_DIR = os.getenv('BASE_DIR', '')
else:
    BASE_DIR = '/Users/saahilagrawal/Documents/personal/projects/'
    os.environ['BASE_DIR'] = BASE_DIR
    # raise Warning("Please set the directory which contains the github repo as the base directory by doing: "
    #               "export $BASE_DIR=<~/path/to/repo>")

RUNTIME_DATA_DIR_NAME = 'agent_data'
REPO_BASE_DIR = 'pubmed_qa_bot'
OTHER_DATA_DIR_NAME = 'pubmed_data'

PAPER_EMBEDDING_DIR = 'paper_embedding'
SCANN_DB_DIR = "trained_scann"
TRAINED_ANNOY_DIR = "trained_annoy"
ANNOY_FILE_NAME = "index.ann"
EMBEDDING_IDX_TO_PAPER_MAP = 'embedding_idx_paper_file_name_map.jsonl'

RUNTIME_DATA_DIR_PATH = os.path.join(BASE_DIR, REPO_BASE_DIR, RUNTIME_DATA_DIR_NAME)
EMBEDDINGS_FOLDER_PATH = os.path.join(RUNTIME_DATA_DIR_PATH, PAPER_EMBEDDING_DIR)
TRAINED_VECTOR_DB_PATH = os.path.join(RUNTIME_DATA_DIR_PATH, SCANN_DB_DIR)
TRAINED_ANNOY_DB_PATH = os.path.join(RUNTIME_DATA_DIR_PATH, TRAINED_ANNOY_DIR)
EMBEDDING_INDEX_TO_PAPER_FILE_PATH = os.path.join(RUNTIME_DATA_DIR_PATH, EMBEDDING_IDX_TO_PAPER_MAP)

OTHER_DATA_DIR = os.path.join(BASE_DIR, OTHER_DATA_DIR_NAME)
ABSTRACT_EMBEDDINGS_FILE_PATH = EMBEDDINGS_FOLDER_PATH + '/split_{}_{}.jsonl'
BODY_EMBEDDINGS_FILE_PATH = os.path.join(OTHER_DATA_DIR, 'body_embeddings', 'split_{}_{}.jsonl')
CLEANED_TEXT_FOLDER_PATH = os.path.join(RUNTIME_DATA_DIR_PATH, 'cleaned_text')