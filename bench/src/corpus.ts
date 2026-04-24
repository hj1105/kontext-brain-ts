/**
 * Tech documentation corpus for benchmarking.
 * 12 documents across backend, frontend, ops, security domains.
 */

export interface BenchDoc {
  id: string;
  title: string;
  body: string;
}

export const CORPUS: BenchDoc[] = [
  {
    id: "backend-rest",
    title: "REST API Design Principles",
    body: `REST APIs should use nouns for resources and HTTP verbs for actions. Use GET for retrieval, POST for creation, PUT for full updates, PATCH for partial updates, and DELETE for removal. Return standard HTTP status codes: 200 OK, 201 Created, 400 Bad Request, 401 Unauthorized, 404 Not Found, 500 Internal Server Error. Version your API in the URL path like /v1/users. Use plural nouns for collections.`,
  },
  {
    id: "backend-postgres",
    title: "PostgreSQL Schema Design",
    body: `PostgreSQL schema design: use UUIDs for primary keys in distributed systems, timestamps with time zone (TIMESTAMPTZ) for all temporal fields, and JSONB for semi-structured data. Normalize to 3NF unless performance requires denormalization. Create indexes on foreign keys and columns used in WHERE clauses. Use partial indexes for filtered queries. Avoid storing large binary data in the database.`,
  },
  {
    id: "backend-auth",
    title: "JWT Authentication Best Practices",
    body: `JWT authentication: store the signing secret in an environment variable, never commit it to source control. Use HS256 or RS256 algorithms. Set short expiration times (15 minutes for access tokens) and use refresh tokens stored in httpOnly cookies. Include user ID, role, and issued-at claims. Rotate signing keys periodically. Never store sensitive data like passwords in JWT payload.`,
  },
  {
    id: "frontend-react",
    title: "React Component Patterns",
    body: `React component patterns: prefer functional components with hooks. Use useState for local state, useEffect for side effects with proper dependency arrays, and useMemo/useCallback to prevent unnecessary re-renders. Extract custom hooks when logic is reused across components. Use React.memo for pure components. Keep components small and single-purpose. Co-locate styles with components using CSS modules or styled-components.`,
  },
  {
    id: "frontend-typescript",
    title: "TypeScript in React Projects",
    body: `TypeScript in React: define prop types with interfaces, not types, for extensibility. Use discriminated unions for state machines. Avoid 'any'; prefer 'unknown' when the type is truly unknown. Use satisfies operator for const assertions. Enable strict mode including noUncheckedIndexedAccess. Type your API responses with zod or similar runtime validators. Export types alongside components.`,
  },
  {
    id: "frontend-perf",
    title: "Frontend Performance Optimization",
    body: `Frontend performance: code-split routes with dynamic imports. Lazy-load images with loading="lazy". Preload critical resources with link rel=preload. Minimize bundle size with tree shaking. Use Web Vitals to measure: LCP under 2.5s, FID under 100ms, CLS under 0.1. Compress images with WebP/AVIF. Serve static assets from a CDN. Defer non-critical JavaScript.`,
  },
  {
    id: "ops-docker",
    title: "Docker Best Practices",
    body: `Docker best practices: use multi-stage builds to minimize image size. Base on alpine or distroless images for security. Never run containers as root; use USER directive. Copy only what you need; use .dockerignore. Pin image versions with specific tags, not latest. Cache dependencies in separate layers. Set HEALTHCHECK directives. Use BuildKit for faster builds. Scan images for vulnerabilities.`,
  },
  {
    id: "ops-kubernetes",
    title: "Kubernetes Deployment Patterns",
    body: `Kubernetes deployment: use Deployments for stateless apps, StatefulSets for stateful ones, DaemonSets for node-local services. Set resource requests and limits on every container. Use liveness and readiness probes. Implement graceful shutdown with preStop hooks. Use ConfigMaps and Secrets for configuration. Apply PodDisruptionBudgets for availability. Use Horizontal Pod Autoscaler based on metrics.`,
  },
  {
    id: "ops-cicd",
    title: "CI/CD Pipeline Design",
    body: `CI/CD pipeline: run tests on every pull request before merge. Lint, type-check, and test in parallel stages. Build once, deploy anywhere: produce immutable artifacts. Use trunk-based development with short-lived feature branches. Automate database migrations with versioned SQL scripts. Deploy to staging automatically, production with manual approval. Monitor deployment health with canary releases.`,
  },
  {
    id: "sec-owasp",
    title: "OWASP Top 10 Security Risks",
    body: `OWASP Top 10: injection attacks (SQL, NoSQL, command) - use parameterized queries. Broken authentication - enforce strong password policies, MFA. Sensitive data exposure - encrypt data at rest and in transit with TLS 1.3. XML external entities - disable external entity processing. Broken access control - enforce authorization on every request. Security misconfiguration - disable default accounts, keep software updated.`,
  },
  {
    id: "sec-secrets",
    title: "Secret Management",
    body: `Secret management: never commit secrets to source control. Use a secret manager like AWS Secrets Manager, HashiCorp Vault, or Azure Key Vault. Rotate secrets regularly, at minimum every 90 days. Use short-lived credentials when possible. Scan repositories for accidentally committed secrets with tools like gitleaks or trufflehog. Use environment variables for local development only. Audit secret access.`,
  },
  {
    id: "sec-tls",
    title: "TLS and HTTPS Configuration",
    body: `TLS configuration: use TLS 1.3, disable TLS 1.0 and 1.1. Configure HSTS with preload to prevent downgrade attacks. Use certificates from a trusted CA like Let's Encrypt. Implement OCSP stapling. Disable weak cipher suites; prefer AEAD ciphers like AES-GCM and ChaCha20-Poly1305. Use certificate pinning for mobile apps. Automate certificate renewal. Monitor certificate expiration.`,
  },
  // ── extended corpus (added to stress-test BM25/MMR on a larger doc set) ──
  { id: "backend-graphql", title: "GraphQL Schema Design", body: `GraphQL schemas should expose types matching domain models, not database tables. Use scalars (ID, String, Int) for primitives. Define input types for mutations separately. Avoid over-fetching with fragments. Use DataLoader to batch N+1 queries. Version by deprecating fields, not URL versioning. Use directives for auth.` },
  { id: "backend-grpc", title: "gRPC Service Patterns", body: `gRPC uses protobuf for schema. Use unary RPCs for request/response, server streaming for subscriptions. Define services in .proto files with explicit field numbers. Use deadlines on every call. Apply interceptors for auth, logging, retry. Generate stubs in multiple languages.` },
  { id: "backend-cache", title: "Redis Caching Patterns", body: `Use Cache-Aside for read-heavy workloads. Set TTLs on every key. Use Redis Cluster for sharding above ~25GB. Pipeline commands to reduce round-trips. Use Lua scripts for atomic multi-step ops. Pick eviction policy (allkeys-lru, volatile-lru) based on use case.` },
  { id: "backend-queue", title: "Message Queue Selection", body: `RabbitMQ for traditional work queues with rich routing. Kafka for high-throughput event streaming and replayable logs. SQS for managed simple queues on AWS. For exactly-once semantics use Kafka transactions or idempotent producers. Always implement dead-letter queues. Set message TTLs.` },
  { id: "backend-rate-limit", title: "Rate Limiting Strategies", body: `Algorithms: token bucket allows bursts, leaky bucket smooths traffic, fixed window is simple, sliding window is accurate. Implement per-user, per-IP, per-endpoint limits. Use Redis with INCR + EXPIRE for distributed rate limiting. Return 429 Too Many Requests with Retry-After header.` },
  { id: "frontend-state", title: "State Management in React", body: `For local state use useState. For shared state across siblings, lift state up or use React Context. For complex global state use Redux Toolkit, Zustand, or Jotai. Server state belongs in TanStack Query or SWR, not in client state stores. URL state lives in the router.` },
  { id: "frontend-forms", title: "Form Handling Best Practices", body: `Use React Hook Form or Formik for complex forms. Validate on both client (UX) and server (security). Show inline error messages near the field. Disable submit during submission to prevent double-submits. Auto-save drafts. Use HTML5 input types for built-in validation.` },
  { id: "frontend-a11y", title: "Accessibility a11y Essentials", body: `Use semantic HTML elements (button, nav, main) instead of div soup. Provide alt text for images. Ensure keyboard navigation works (focus order, focus trap in modals). Maintain color contrast ratio at least 4.5:1 for body text. Test with screen readers (NVDA, VoiceOver). Respect prefers-reduced-motion.` },
  { id: "frontend-i18n", title: "Internationalization i18n Setup", body: `Use i18next or react-intl for translations. Externalize all user-facing strings into JSON files per locale. Use ICU MessageFormat for plurals. Right-to-left languages need bidirectional text support. Format dates and numbers per locale with Intl APIs. Lazy-load locale bundles.` },
  { id: "ops-monitor", title: "Observability Metrics Logs Traces", body: `Three pillars of observability: metrics (Prometheus, Datadog), logs (Loki, ELK), traces (Jaeger, Tempo). Use structured logging in JSON for searchability. Log levels: DEBUG, INFO, WARN, ERROR, FATAL. Use trace IDs to correlate across services. RED metrics for services. USE metrics for resources. Alert on symptoms, not causes.` },
  { id: "ops-iac", title: "Infrastructure as Code Terraform", body: `Terraform manages cloud resources declaratively. Store state in remote backend (S3 + DynamoDB lock). Use modules for reusable patterns. Workspace per environment (dev, staging, prod). Plan before apply. Use terraform fmt and tflint. Avoid manual changes outside Terraform. Tag every resource for cost tracking.` },
  { id: "ops-backup", title: "Backup and Disaster Recovery", body: `3-2-1 backup rule: 3 copies, 2 different media, 1 offsite. Test restores monthly — untested backups don't exist. Define RPO and RTO per system. Automate snapshots. Encrypt backups at rest and in transit. Document recovery runbook. Practice with chaos engineering.` },
  { id: "sec-rbac", title: "Role-Based Access Control RBAC", body: `RBAC defines permissions via roles, not individual users. Principle of least privilege: grant minimum permissions needed. Separate read, write, admin roles. Audit role assignments quarterly. Avoid wildcard permissions. Use groups in IAM (AWS, GCP, Azure) for assignment. Implement break-glass admin accounts.` },
  { id: "sec-csrf", title: "CSRF and XSS Defenses", body: `CSRF: use SameSite=Strict cookies, double-submit tokens, or origin header checks. XSS: escape user input in HTML context, use Content-Security-Policy header to restrict script sources. Avoid innerHTML; prefer textContent. Sanitize markdown server-side. Set HttpOnly on session cookies.` },
  { id: "sec-supply-chain", title: "Software Supply Chain Security", body: `Pin dependencies to exact versions (package-lock.json). Use SBOM tools like Syft. Scan dependencies for CVEs with Snyk, Dependabot, or Trivy. Verify package integrity with hashes. Use private registries for internal packages. Avoid postinstall scripts from untrusted sources. Sign commits with GPG or Sigstore.` },
  { id: "data-etl", title: "ETL Pipeline Design", body: `Prefer ELT (load then transform in warehouse) over ETL when warehouse is cheap. Use dbt for SQL transformations. Schedule with Airflow, Dagster, or Prefect. Make pipelines idempotent. Implement late-arriving data handling. Monitor freshness and row counts. Version transformations alongside code.` },
  { id: "data-warehouse", title: "Data Warehouse Modeling", body: `Star schema: facts in center, dimensions around. Snowflake schema normalizes dimensions further. SCDs: type 1 overwrite, type 2 history, type 3 limited history. Use surrogate keys for joins. Partition large fact tables by date. Cluster on frequently-filtered columns.` },
  { id: "ml-features", title: "ML Feature Store Patterns", body: `A feature store decouples feature engineering from model training and serving. Provides offline store (training) and online store (serving) with consistency. Use Feast or Tecton. Define features once; reuse across models. Track feature lineage. Monitor feature drift. Minimize training-serving skew with point-in-time correct features.` },
];

export interface BenchQuery {
  id: string;
  question: string;
  /** Document IDs that should ideally be retrieved for this question. */
  expectedDocIds: string[];
  /** Short expected answer fragments (any-match = correct). */
  expectedKeywords: string[];
}

export const QUERIES: BenchQuery[] = [
  {
    id: "q1",
    question: "How should I version my REST API?",
    expectedDocIds: ["backend-rest"],
    expectedKeywords: ["v1", "URL", "path"],
  },
  {
    id: "q2",
    question: "What expiration time is appropriate for JWT access tokens?",
    expectedDocIds: ["backend-auth"],
    expectedKeywords: ["15 minutes", "short"],
  },
  {
    id: "q3",
    question: "Should I use any type in TypeScript?",
    expectedDocIds: ["frontend-typescript"],
    expectedKeywords: ["unknown", "avoid"],
  },
  {
    id: "q4",
    question: "How do I keep Docker images small?",
    expectedDocIds: ["ops-docker"],
    expectedKeywords: ["multi-stage", "alpine", "distroless"],
  },
  {
    id: "q5",
    question: "How should I store and rotate application secrets?",
    expectedDocIds: ["sec-secrets"],
    expectedKeywords: ["secret manager", "Vault", "rotate", "90"],
  },
  {
    id: "q6",
    question: "What TLS version should I use in production?",
    expectedDocIds: ["sec-tls"],
    expectedKeywords: ["1.3", "TLS"],
  },
  {
    id: "q7",
    question: "How can I prevent unnecessary React re-renders?",
    expectedDocIds: ["frontend-react"],
    expectedKeywords: ["useMemo", "useCallback", "memo"],
  },
  {
    id: "q8",
    question: "What Web Vitals thresholds matter for performance?",
    expectedDocIds: ["frontend-perf"],
    expectedKeywords: ["LCP", "FID", "CLS", "2.5", "100"],
  },
  // ── extended queries to stress new variants on the larger corpus ──
  {
    id: "q9",
    question: "When should I use Kafka instead of RabbitMQ?",
    expectedDocIds: ["backend-queue"],
    expectedKeywords: ["high-throughput", "event streaming", "replayable"],
  },
  {
    id: "q10",
    question: "How do I prevent N+1 queries in GraphQL?",
    expectedDocIds: ["backend-graphql"],
    expectedKeywords: ["DataLoader", "batch"],
  },
  {
    id: "q11",
    question: "What's the difference between RED and USE metrics?",
    expectedDocIds: ["ops-monitor"],
    expectedKeywords: ["Rate", "Errors", "Duration", "Utilization", "Saturation"],
  },
  {
    id: "q12",
    question: "How should I model slowly changing dimensions in a data warehouse?",
    expectedDocIds: ["data-warehouse"],
    expectedKeywords: ["type 1", "type 2", "history", "overwrite"],
  },
];
