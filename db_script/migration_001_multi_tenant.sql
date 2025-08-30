-- Migration 001: Multi-tenant SaaS refactoring
-- Add tenants, users tables and tenant isolation using Row-Level Security (RLS)

-- Create tenants table
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    plan TEXT DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'enterprise')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT, -- Nullable for OAuth/SSO users
    role TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'member')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add tenant_id and created_by to existing tables
ALTER TABLE documents ADD COLUMN tenant_id UUID;
ALTER TABLE documents ADD COLUMN created_by UUID;
ALTER TABLE chunks ADD COLUMN tenant_id UUID;

-- Create Default Tenant for existing data
INSERT INTO tenants (id, name, slug, plan) 
VALUES ('00000000-0000-0000-0000-000000000001', 'Default Tenant', 'default', 'enterprise');

-- Backfill existing documents and chunks with Default Tenant
UPDATE documents SET tenant_id = '00000000-0000-0000-0000-000000000001' WHERE tenant_id IS NULL;
UPDATE chunks SET tenant_id = '00000000-0000-0000-0000-000000000001' WHERE tenant_id IS NULL;

-- Make tenant_id NOT NULL after backfill
ALTER TABLE documents ALTER COLUMN tenant_id SET NOT NULL;
ALTER TABLE chunks ALTER COLUMN tenant_id SET NOT NULL;

-- Add foreign key constraints
ALTER TABLE documents ADD CONSTRAINT fk_documents_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE;
ALTER TABLE documents ADD CONSTRAINT fk_documents_created_by FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL;
ALTER TABLE chunks ADD CONSTRAINT fk_chunks_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE;

-- Create indexes for tenant isolation performance
CREATE INDEX idx_tenants_slug ON tenants(slug);
CREATE INDEX idx_users_tenant_id ON users(tenant_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_documents_tenant_id ON documents(tenant_id);
CREATE INDEX idx_documents_tenant_created_at ON documents(tenant_id, created_at);
CREATE INDEX idx_chunks_tenant_id ON chunks(tenant_id);
CREATE INDEX idx_chunks_tenant_doc_id ON chunks(tenant_id, doc_id);

-- Enable Row Level Security (RLS)
ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

-- RLS Policies for tenant isolation

-- Tenants: Users can only see their own tenant
CREATE POLICY tenant_isolation ON tenants
    FOR ALL
    USING (id = COALESCE(current_setting('app.tenant_id', true)::UUID, id));

-- Users: Users can only see users from their tenant
CREATE POLICY user_tenant_isolation ON users
    FOR ALL
    USING (tenant_id = COALESCE(current_setting('app.tenant_id', true)::UUID, tenant_id));

-- Documents: Users can only see documents from their tenant
CREATE POLICY document_tenant_isolation ON documents
    FOR ALL
    USING (tenant_id = COALESCE(current_setting('app.tenant_id', true)::UUID, tenant_id));

-- Chunks: Users can only see chunks from their tenant
CREATE POLICY chunk_tenant_isolation ON chunks
    FOR ALL
    USING (tenant_id = COALESCE(current_setting('app.tenant_id', true)::UUID, tenant_id));

-- Update existing functions to be tenant-aware

-- Tenant-aware similarity search function
CREATE OR REPLACE FUNCTION search_similar_chunks_tenant(
    query_embedding VECTOR(1536),
    tenant_id_param UUID,
    similarity_threshold FLOAT DEFAULT 0.0,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE(
    chunk_id TEXT,
    doc_id UUID,
    filename TEXT,
    page INTEGER,
    chunk_idx INTEGER,
    content TEXT,
    content_tokens INTEGER,
    created_at TIMESTAMP WITH TIME ZONE,
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.chunk_id,
        c.doc_id,
        c.filename,
        c.page,
        c.chunk_idx,
        c.content,
        c.content_tokens,
        c.created_at,
        1 - (c.embedding <=> query_embedding) AS similarity_score
    FROM chunks c
    WHERE c.tenant_id = tenant_id_param
      AND (1 - (c.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Tenant-aware document exists check
CREATE OR REPLACE FUNCTION document_exists_by_hash_tenant(
    content_hash_param TEXT,
    tenant_id_param UUID
)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS(
        SELECT 1 FROM documents 
        WHERE content_hash = content_hash_param 
          AND tenant_id = tenant_id_param
    );
END;
$$ LANGUAGE plpgsql;

-- Tenant-aware stats function
CREATE OR REPLACE FUNCTION get_database_stats_tenant(tenant_id_param UUID)
RETURNS TABLE(
    document_count BIGINT,
    chunk_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM documents WHERE tenant_id = tenant_id_param) as document_count,
        (SELECT COUNT(*) FROM chunks WHERE tenant_id = tenant_id_param) as chunk_count;
END;
$$ LANGUAGE plpgsql;

-- Create usage metering table
CREATE TABLE tenant_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    requests_count INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    storage_bytes BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(tenant_id, date)
);

CREATE INDEX idx_tenant_usage_tenant_date ON tenant_usage(tenant_id, date);

-- Enable RLS on usage table
ALTER TABLE tenant_usage ENABLE ROW LEVEL SECURITY;

CREATE POLICY usage_tenant_isolation ON tenant_usage
    FOR ALL
    USING (tenant_id = COALESCE(current_setting('app.tenant_id', true)::UUID, tenant_id));