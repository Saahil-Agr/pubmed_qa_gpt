# pubmed_qa_gpt
A Question Answer agent specifically focussed on answering questions based on information in pubmed open source papers.

## Evnironment Setup 
virtualenv -p python3.8 venv
pip install -r requirements.txt

# Interactive Chat Bot Usage
python $BASE_DIR/pubmed_qa_bot/pubmed_agent_cli.py \
      --api_key dummy_key
      --state 0 \
      --question "What is the role of chemokines and chemokine receptors by lymphatic vessels (LVs) in leukocyte traffic and cancer metastasis?"

## Various Indpendent Components
1. Code to generate the embdeddings form the cleaned text of papers that is split into Title + Abstract and Body
2. Building a vector DB that stores the embeddings and finds the nearest neighbor when a new embedding is provided
3. DocumentRetriever class that nicely wraps around the different use cases of generating embeddings for new NLP qury, fetching the candidate documents and serving the text for those documents.
4. A chat bot that relies on DocumentRetriever and the OpenAI GPT3.5 model to reason about the retrieved documents.
