from annoy import AnnoyIndex
import attr
import os
import json
import constants
EMBEDDING_VECTOR_DIM = 768


@attr.s
class ScannConfig:
    distance_method = attr.ib(default='dot_product')
    should_partition = attr.ib(default=False)
    num_leaves = attr.ib(default=2000)
    scoring_method = attr.ib(default='brute_force')  # Ideally this should of enum class
    num_neighbors = attr.ib(default=10)

@attr.s
class AnnoyConfig:
    distance_method = attr.ib(default='angular')
    num_trees = attr.ib(default=1000)
    seed = attr.ib(default=44)  # Ideally this should of enum class
    num_neighbors = attr.ib(default=10)
    file_name = attr.ib(default=constants.ANNOY_FILE_NAME)


# def create_and_train_scann_vector_db(embeddings_array, scann_config: VectorDbConfig):
#     # TODO: Allow better configurability in instantiation. For now using a bunch of preset values
#     if scann_config.scoring_method == 'brute_force':
#         searcher = scann.scann_ops_pybind.builder(embeddings_array, scann_config.num_neighbors,
#                                                   "dot_product").score_brute_force().build()
#     else:
#         if scann_config.should_partition:
#             searcher = scann.scann_ops_pybind.builder(embeddings_array, scann_config.num_neighbors, "dot_product"). \
#                 score_ah(2, anisotropic_quantization_threshold=0.2). \
#                 reorder(100).build(). \
#                 tree(num_leaves=scann_config.num_leaves, num_leaves_to_search=100, training_sample_size=10000)
#         else:
#             searcher = scann.scann_ops_pybind.builder(embeddings_array, scann_config.num_neighbors, "dot_product"). \
#                 score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build()
#
#     return searcher

def create_and_train_annoy_vector_db(embeddings_array, annoy_config: AnnoyConfig):
    # TODO: Allow better configurability in instantiation. For now using a bunch of preset values
    vector_length = len(embeddings_array[0])
    annoy_index = AnnoyIndex(vector_length, metric=annoy_config.distance_method)
    annoy_index.set_seed(annoy_config.seed)
    for idx, embedding in enumerate(embeddings_array):
        annoy_index.add_item(idx, embedding)
    annoy_index.build(annoy_config.num_trees)
    return annoy_index


def save_serialized_index(vector_db, folder_path, file_name=constants.ANNOY_FILE_NAME):
    os.makedirs(folder_path, exist_ok=True)
    vector_db.save(os.path.join(folder_path, file_name))


def read_stored_embeddings(file_path):
    with open(file_path, 'r') as f:
        paper_embedding_map = json.loads(f.readlines()[0])

    return paper_embedding_map


def create_and_store_embedding_idx_paper_map(embeddings_folder_path, embedding_index_to_paper_file_path):
    all_embeddings = []
    for file_path in os.listdir(embeddings_folder_path):
        with open(os.path.join(embeddings_folder_path,file_path), 'r') as f:
            embeddings = json.loads(f.readlines()[0])

        file_name = '_'.join(file_path.split('/')[-1].split('_')[:-1]) + '.jsonl'
        embeddings_list = list((paper, file_name, embedding) for paper, embedding in embeddings.items())
        all_embeddings.extend(embeddings_list)

    all_embeddings.sort()
    embeddings_idx_to_paper_file_map = {
        idx: (paper_embed[0], paper_embed[1]) for idx, paper_embed in enumerate(all_embeddings)
    }
    try:
        with open(embedding_index_to_paper_file_path, 'x') as f:
            f.write(json.dumps(embeddings_idx_to_paper_file_map))
    except:
        print('file exists skip writing')
    return all_embeddings


if __name__ == "__main__":

    paper_embeddings = create_and_store_embedding_idx_paper_map(
        constants.EMBEDDINGS_FOLDER_PATH, constants.EMBEDDING_INDEX_TO_PAPER_FILE_PATH
    )
    embedding_array = [embed[-1] for embed in paper_embeddings]
    trained_vector_db = create_and_train_annoy_vector_db(embedding_array, AnnoyConfig())
    save_serialized_index(trained_vector_db, constants.TRAINED_ANNOY_DB_PATH)
