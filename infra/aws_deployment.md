# Optional AWS Deployment

This project is designed to run locally first. A production AWS deployment should keep the same boundaries:

- API: ECS Fargate service running `uvicorn api.main:app`.
- Frontend: S3 + CloudFront static build, or ECS if using Vite preview.
- Vector store: managed OpenSearch, Aurora PostgreSQL with pgvector, or a persistent Chroma/Qdrant service.
- Secrets: AWS Secrets Manager or SSM Parameter Store. Do not bake API keys into images.
- Logs: CloudWatch JSON logs from `StructuredLogger`.
- Storage: S3 for uploaded source documents and generated eval/RAFT artifacts.

Minimum hardening before public exposure:

- Add authentication and per-user authorization.
- Put API Gateway or an ALB with rate limits in front of the API.
- Enforce upload size and content-type limits at the edge.
- Move from in-memory retrieval to a persistent vector backend.
