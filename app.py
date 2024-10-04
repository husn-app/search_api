from flask import Flask, jsonify, request
import open_clip
import time
import torch
import torch.nn.functional as F
import faiss
import os

torch.set_grad_enabled(False)

model, tokenizer = None, None
image_embeddings = None
faiss_index = None

DEPLOYMENT_TYPE = os.environ.get('DEPLOYMENT_TYPE', 'LOCAL')

ROOT_DIR = '/husn-cool-storage/' if (DEPLOYMENT_TYPE == 'PROD') else './'

def init_model():
    global model, tokenizer
    print('Initializing model...')
    start_time = time.time()
    
    # initate model.
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.visual = None
    model = torch.compile(model)
    
    # initate tokenizer.
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print('Initialized model.\tTime Taken: ', time.time() - start_time)

def init_image_embeddings():
    global image_embeddings
    print('Reading image embeddings...')
    start_time = time.time()
    image_embeddings = F.normalize(torch.load(ROOT_DIR + 'image_embeddings_kaggle.pt'), dim=-1).detach().numpy()
    print('Read image embeddings.\nTime Taken: ', time.time() - start_time)
    
def init_faiss_index():
    global faiss_index
    faiss_index = faiss.IndexFlatIP(512)
    faiss_index.add(image_embeddings) 

def init_ml():
    init_model()
    init_image_embeddings()
    init_faiss_index()
    

init_ml()
# Assert model
assert model is not None
assert tokenizer is not None 
# Assert embeddings
assert image_embeddings is not None
assert faiss_index is not None

app = Flask(__name__)

# DEPRECATED : Use faiss_index.search instead. 
def getTopK(base_embedding, K=100):
    global image_embeddings
    probs = torch.nn.functional.cosine_similarity(image_embeddings, base_embedding.view(1, 512))
    topk_indices = probs.topk(K).indices
    return topk_indices, probs[topk_indices]

def process_query(query):
    global tokenizer, faiss_index, final_df
    if not query:
        return None, "Query cannot be empty", 400
    
    query_encoding = tokenizer(query)
    query_embedding = F.normalize(model.encode_text(query_encoding), dim=-1) # shape = [1, DIM]

    ## faiss_index.search expects batched inputs.
    topk_scores, topk_indices = faiss_index.search(query_embedding.detach().numpy(),  100) 
    topk_scores, topk_indices = topk_scores[0], topk_indices[0]

    return {"query": query, "indices": topk_indices.tolist(), "scores": topk_scores.tolist()}, None, 200
@app.route('/api/query', methods=['GET'])
def api_query():
    # Try to get query from JSON data first
    data = request.get_json(silent=True)
    if data and 'query' in data:
        query = data.get('query')
    else:
        # If not in JSON, try to get from query parameters
        query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    result, error, status_code = process_query(query)
    if error:
        return jsonify({"error": error}), status_code
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
