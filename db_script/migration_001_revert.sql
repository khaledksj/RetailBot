-- Revert Migration 001: Multi-tenant SaaS refactoring
-- WARNING: This will remove all tenant and user data

-- Disable RLS
ALTER TABLE tenants DISABLE ROW LEVEL SECURITY;
ALTER TABLE users DISABLE ROW LEVEL SECURITY;
ALTER TABLE documents DISABLE ROW LEVEL SECURITY;
ALTER TABLE chunks DISABLE ROW LEVEL SECURITY;

-- Drop policies
DROP POLICY IF EXISTS tenant_isolation ON tenants;
DROP POLICY IF EXISTS user_tenant_isolation ON users;
DROP POLICY IF EXISTS document_tenant_isolation ON documents;
DROP POLICY IF EXISTS chunk_tenant_isolation ON chunks;

-- Drop foreign key constraints
ALTER TABLE documents DROP CONSTRAINT IF EXISTS fk_documents_tenant;
ALTER TABLE documents DROP CONSTRAINT IF EXISTS fk_documents_created_by;
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS fk_chunks_tenant;

-- Remove added columns
ALTER TABLE documents DROP COLUMN IF EXISTS tenant_id;
ALTER TABLE documents DROP COLUMN IF EXISTS created_by;
ALTER TABLE chunks DROP COLUMN IF EXISTS tenant_id;

-- Drop new tables
DROP TABLE IF EXISTS tenant_usage;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS tenants;

-- Drop tenant-aware functions
DROP FUNCTION IF EXISTS search_similar_chunks_tenant(VECTOR(1536), UUID, FLOAT, INTEGER);
DROP FUNCTION IF EXISTS document_exists_by_hash_tenant(TEXT, UUID);
DROP FUNCTION IF EXISTS get_database_stats_tenant(UUID);