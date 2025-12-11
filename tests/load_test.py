from locust import HttpUser, task, between

class RAGLoadTest(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        self.token = "your_jwt_token_here"
    
    @task
    def query_rag(self):
        self.client.post(
            "/query",
            json={
                "question": "What is machine learning?",
                "use_hybrid": True,
                "use_multi_query": True
            },
            headers={"Authorization": f"Bearer {self.token}"}
        )
    
    @task
    def upload_document(self):
        # Simulate file upload
        self.client.post(
            "/documents/upload",
            files={"file": ("test.pdf", b"dummy content", "application/pdf")},
            headers={"Authorization": f"Bearer {self.token}"}
        )