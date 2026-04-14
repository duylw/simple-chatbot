import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness

# Import your services (assuming you run this script with access to the DB/Vector stores)
from src.services.rag.bm25 import make_bm25_retriever
from src.services.rag.vectordb import make_vector_db_retriever
from src.services.rag.factory import make_agentic_rag_service

def calculate_retriever_metrics(retrieved_docs, ground_truth_docs, k):
    """Calculates Precision@k and Recall@k assuming string matching for simplicity."""
    retrieved_texts = [doc.page_content for doc in retrieved_docs[:k]]
    
    # Simple substring matching. For production, match Chunk IDs if possible.
    relevant_retrieved = 0
    for gt_doc in ground_truth_docs:
        if any(gt_doc in retr_doc for retr_doc in retrieved_texts):
             relevant_retrieved += 1

    precision = relevant_retrieved / k
    recall = relevant_retrieved / len(ground_truth_docs) if len(ground_truth_docs) > 0 else 0
    return precision, recall


async def run_evaluation(eval_data: dict, k=5):
    # 1. Initialize your RAG system
    print("Initializing Retrievers...")
    bm25_retriever = make_bm25_retriever()
    chroma_retriever = make_vector_db_retriever()
    rag_service = make_agentic_rag_service(bm25_retriever, chroma_retriever, retriever_top_k=k)

    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": eval_data["ground_truth"]
    }
    
    total_precision = 0
    total_recall = 0

    print("Running system on evaluation dataset...")
    # 2. Run queries through the system
    for i, question in enumerate(eval_data["question"]):
        # Run the full agentic graph
        response = await rag_service.ask(question)
        
        # Store for RAGAS
        results["question"].append(question)
        results["answer"].append(response["answer"])
        results["contexts"].append([doc.page_content for doc in response["sources"]])
        
        # Calculate Retriever Metrics (Precision & Recall)
        p, r = calculate_retriever_metrics(
            response["sources"], 
            eval_data["ground_truth_context"][i], 
            k
        )
        total_precision += p
        total_recall += r
        print(f"Query {i+1}: Precision@{k}: {p:.2f}, Recall@{k}: {r:.2f}")

    # 3. Final Retriever Metrics
    n = len(eval_data["question"])
    print("\n--- Retriever Results ---")
    print(f"Average Precision@{k}: {total_precision/n:.2f}")
    print(f"Average Recall@{k}: {total_recall/n:.2f}")

    # 4. Run RAGAS Generator Evaluation
    print("\n--- Running RAGAS Evaluation ---")
    dataset = Dataset.from_dict(results)
    
    ragas_results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness]
    )
    
    print("\n--- RAGAS Generator Results ---")
    print(ragas_results)

# To run this script:
# asyncio.run(run_evaluation(evaluation_data, k=10))