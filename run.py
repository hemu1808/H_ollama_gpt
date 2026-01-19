#!/usr/bin/env python3
import argparse
import uvicorn
from config import settings

def main():
    parser = argparse.ArgumentParser(description="Production RAG System")
    parser.add_argument("--mode", choices=["api", "worker", "all"], default="api")
    args = parser.parse_args()
    
    if args.mode in ["api", "all"]:
        uvicorn.run(
            "api_main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            workers=settings.WORKERS,
            log_level="info"
        )
    
    if args.mode in ["worker", "all"]:
        from celery import Celery
        from workers.celery_worker import celery_app
        
        worker = celery_app.Worker(
            concurrency=4,
            loglevel="info"
        )
        worker.start()

if __name__ == "__main__":
    main()