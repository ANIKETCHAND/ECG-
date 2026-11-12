# Deployment, High Availability, & Disaster Recovery Plan

## 1. Production Deployment Topology

ECG Guardian is structured for cloud-native deployment leveraging high-availability managed infrastructure:

- **Compute Layer**: Vercel Serverless Functions (`api/index.py` via Python 3.11+ runtime).
  - Geographically distributed edge network.
  - Automatic scaling from 0 to peak concurrency.
  - Ephemeral containers with zero local state dependencies.
- **Relational Persistence Layer**: Supabase PostgreSQL 15+ (AWS region matching compute).
  - High availability multi-AZ replica deployment.
  - Automatic connection pooling (PgBouncer) handling up to 1,000 concurrent client sessions.
- **Storage Layer**: AWS S3 / Supabase Storage with AES-256 server-side encryption.
  - Private clinical object bucket (`ecg-studies`, `ecg-reports`).
  - Short-lived signed URLs (15-minute expiration) generated only after session authorization.
- **Client Tier**: Modern HTML5 / Vanilla ES6+ SPA served via global Vercel CDN.

---

## 2. Environment Configuration & Secret Management

All secrets must be injected into the Vercel production runtime environment and must **never** be committed to source control:

| Variable Name | Description | Environment Scope | Example Format |
| :--- | :--- | :--- | :--- |
| `JWT_SECRET` | Cryptographic secret for signing session JWT tokens | Production, Preview | 64-character hex string |
| `ECG_STRICT_AUTH` | Enforces zero-trust authentication on all clinical endpoints | Production | `1` |
| `SUPABASE_URL` | Canonical PostgreSQL database endpoint | Production, Preview | `https://xxxx.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Administrative service-role key for backend operations | Production | Secure secret string |
| `AWS_S3_BUCKET` | Private storage bucket for source waveforms & PDFs | Production | `ecg-guardian-clinical-prod` |
| `GOOGLE_CLIENT_ID` | OAuth2 Client ID for Google Identity sign-in | Production | `*.apps.googleusercontent.com` |

---

## 3. Disaster Recovery (DR) & Business Continuity Plan

### 3.1 Recovery Objectives
- **Recovery Point Objective (RPO)**: <= 5 minutes (via PostgreSQL continuous WAL archiving).
- **Recovery Time Objective (RTO)**: <= 15 minutes (via automated container redeployment and Supabase failover).

### 3.2 Dual-Persistence Fallback Protocol
In the event of a catastrophic public cloud network outage:
1. The backend automatically switches to local relational persistence (`src/database/db_manager.py`).
2. Clinical staff can continue recording vitals, analyzing emergency ECGs, and generating preliminary findings locally.
3. Once cloud connectivity is re-established, the backend sync service queues and replays committed local transactions to Supabase PostgreSQL, ensuring zero record loss.

### 3.3 Database Backup & Restore Procedures
- **Daily Automated Backups**: Retained for 30 days in off-site geo-redundant storage.
- **Point-in-Time Recovery (PITR)**: Enables millisecond-level rollbacks to recover from accidental corruption or administrative error.
- **Restoration Validation**: Backup restoration drills are conducted semi-annually into a dedicated staging sandbox.

---

## 4. Telemetry, Health Checks, & Incident Response

- **Operational Health Endpoint**: `GET /api/health`
  - Validates API responsiveness, active model status (`ECG-RF-2.0.0-PROD`), signal validation gate status, and database connectivity.
- **Audit Alarm Monitoring**:
  - Failed authentication attempts > 10/min from a single IP triggers automated rate-limiting.
  - Unauthorized attempts to seal clinical reports (`HTTP 403`) trigger high-priority alerts to the Hospital Security Officer.
