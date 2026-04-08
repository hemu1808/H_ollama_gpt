with open("error.log", "w") as f:
    import traceback
    try:
        from rag_service import RAGService
        r = RAGService()
    except Exception as e:
        traceback.print_exc(file=f)
