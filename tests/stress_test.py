import asyncio
import aiohttp
import time
import json
import statistics

# Configuration
API_URL = "http://localhost:8000/query"
CONCURRENCY = 10
TOTAL_REQUESTS = 50

# Test Queries to trigger different aspects of the hybrid algorithm
QUERIES = [
    {"question": "What is ChromaDB?", "mode": "fast", "top_k": 5},
    {"question": "Explain the architecture of HGPT.", "mode": "deep", "top_k": 10},
    {"question": "How does PolarQuant compress embeddings?", "mode": "agentic", "top_k": 3},
    {"question": "Who built this application?", "mode": "fast", "top_k": 5},
    {"question": "What is the relationship between node A and node B?", "mode": "graph", "top_k": 5}
]

async def fire_request(session, query, request_id):
    start_time = time.time()
    try:
        async with session.post(API_URL, json=query) as response:
            if response.status != 200:
                print(f"[Req {request_id}] Failed with status: {response.status}")
                return False, 0.0
            data = await response.json()
            latency = time.time() - start_time
            # print(f"[Req {request_id}] Success | Mode: {query['mode']} | Latency: {latency:.3f}s")
            return True, latency
    except Exception as e:
        print(f"[Req {request_id}] Error: {e}")
        return False, 0.0

async def worker(session, tasks):
    results = []
    for query, req_id in tasks:
        results.append(await fire_request(session, query, req_id))
    return results

async def main():
    print(f"--- Starting Stress Test ---")
    print(f"Target: {API_URL}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Total Requests: {TOTAL_REQUESTS}\n")
    
    # Generate task list
    tasks = []
    for i in range(TOTAL_REQUESTS):
        tasks.append((QUERIES[i % len(QUERIES)], i+1))
        
    # Chunk tasks for workers
    chunk_size = len(tasks) // CONCURRENCY + (1 if len(tasks) % CONCURRENCY != 0 else 0)
    task_chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    
    start_total = time.time()
    
    async with aiohttp.ClientSession() as session:
        worker_coroutines = [worker(session, chunk) for chunk in task_chunks]
        all_results = await asyncio.gather(*worker_coroutines)
        
    total_time = time.time() - start_total
    
    # Flatten results
    flat_results = [item for sublist in all_results for item in sublist]
    successful = [lat for success, lat in flat_results if success]
    failed = len(flat_results) - len(successful)
    
    print(f"\n--- Stress Test Results ---")
    print(f"Total Time        : {total_time:.2f}s")
    print(f"Successful Req    : {len(successful)}")
    print(f"Failed Req        : {failed}")
    if successful:
        print(f"Throughput        : {len(successful) / total_time:.2f} req/s")
        print(f"Min Latency       : {min(successful):.3f}s")
        print(f"Max Latency       : {max(successful):.3f}s")
        print(f"Avg Latency       : {statistics.mean(successful):.3f}s")
        print(f"Median Latency    : {statistics.median(successful):.3f}s")
        
if __name__ == "__main__":
    asyncio.run(main())
