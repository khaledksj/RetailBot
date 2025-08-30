#!/usr/bin/env python3
"""
Backfill script to assign existing documents and chunks to the correct tenant.
Run this after the migration to set proper tenant_id for legacy data.
"""

import asyncio
import asyncpg
import os
import sys
from uuid import UUID

DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"

async def backfill_tenant_data():
    """Backfill existing documents and chunks with Default Tenant ID."""
    
    # Get database connection string
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    conn = None
    try:
        # Connect to database
        conn = await asyncpg.connect(database_url)
        
        # Check if migration has been applied
        try:
            result = await conn.fetchval("SELECT COUNT(*) FROM tenants WHERE id = $1", UUID(DEFAULT_TENANT_ID))
            if result == 0:
                print("ERROR: Default tenant not found. Run migration first.")
                return
        except Exception as e:
            print(f"ERROR: Migration not applied yet. {e}")
            return
        
        # Count existing documents and chunks without tenant_id
        docs_without_tenant = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE tenant_id IS NULL"
        )
        chunks_without_tenant = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE tenant_id IS NULL"
        )
        
        print(f"Found {docs_without_tenant} documents and {chunks_without_tenant} chunks without tenant_id")
        
        if docs_without_tenant == 0 and chunks_without_tenant == 0:
            print("All data already has tenant_id assigned. Nothing to backfill.")
            return
        
        # Start transaction
        async with conn.transaction():
            # Backfill documents
            if docs_without_tenant > 0:
                updated_docs = await conn.fetchval(
                    "UPDATE documents SET tenant_id = $1 WHERE tenant_id IS NULL RETURNING COUNT(*)",
                    UUID(DEFAULT_TENANT_ID)
                )
                print(f"Updated {updated_docs} documents with Default Tenant ID")
            
            # Backfill chunks
            if chunks_without_tenant > 0:
                updated_chunks = await conn.fetchval(
                    "UPDATE chunks SET tenant_id = $1 WHERE tenant_id IS NULL RETURNING COUNT(*)",
                    UUID(DEFAULT_TENANT_ID)
                )
                print(f"Updated {updated_chunks} chunks with Default Tenant ID")
        
        print("Backfill completed successfully!")
        
    except Exception as e:
        print(f"ERROR during backfill: {e}")
        sys.exit(1)
    finally:
        if conn:
            await conn.close()

if __name__ == "__main__":
    asyncio.run(backfill_tenant_data())