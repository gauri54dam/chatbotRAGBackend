# ChaBotRAGBackend



This repository contains the code to deploy PDF RAG with agent and without agent both onto Databricks platform.

For this Deployment, the Databricks Mosaic AI platform, capabilities are leveraged like MLFLow for MLOPs, Vector Search for simiarity search. 

- Query search index using **Databricks Vector DB** ensuring efficient storage and retrieval of embeddings 
- **MLFlow MLOps** to log, publish, and monitor fine-tuned RAG model on the databricks platform
- **Unity Catalog** with access to analytics.models schema to publish model from workspace code
- **Serving endpoint** to serve the published model for end users tests
- **DataBricks Apps** - UI


  Overall Architecture Components of Agentic PDF RAG Application for Chat With my Documents-
  1. UI - Databricks Apps
  2. User questions prompting
  3. Vector Search, RAG - Databricks Vector DB
  4. Langgraph Agents - base LLM model - databricks-meta-llama-3-3-70b-instruct
  5. Tool-based Web Browsing Agent (LLM + Serper Dev Tool)  - Incorporating external info to enhance the fact-based document response - (enhance router skips the entire node when not needed, lowers cost, and is faster for simple questions)
  6. LangGraph Agents with memory for multi-turn tracking
 
