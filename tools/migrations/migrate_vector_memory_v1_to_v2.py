#!/usr/bin/env python3
"""
Migration script for vector memory schema v1 → v2.

New fields in v2:
- epistemic_status: "stated" | "inferred" | "verified" | "contradicted" | "retracted"
- confidence: float (0.0-1.0)
- entities: list of entity names
- keywords: list of BM25 keywords
- related_ids: list of related memory IDs
- is_consolidated: bool
- version: int

This script:
1. Creates a backup of the LanceDB table
2. Adds missing columns (if any) with default values
3. Updates existing rows to have default values
4. Sets version=2 for all rows

Usage:
    python migrate_vector_memory_v1_to_v2.py --verify-only
    python migrate_vector_memory_v1_to_v2.py
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path to allow imports from hermes-agent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import lancedb
    import pandas as pd
    import pyarrow as pa
except ImportError as e:
    print(f"Error: Required module missing: {e}")
    print("Please install: pip install lancedb pandas pyarrow")
    sys.exit(1)

# Default paths
HERMES_HOME = Path.home() / ".hermes"
VECTOR_MEMORY_PATH = HERMES_HOME / "vector_memory"
TABLE_NAME = "memory_vectors"

# v2 schema defaults
EPISTEMIC_STATUS_DEFAULT = "stated"
CONFIDENCE_DEFAULT = 0.5
ENTITIES_DEFAULT = []
KEYWORDS_DEFAULT = []
RELATED_IDS_DEFAULT = []
IS_CONSOLIDATED_DEFAULT = False
VERSION_DEFAULT = 1  # will be updated to 2 after migration


def get_table_schema(table):
    """Return schema as a dict of column name → type."""
    schema = {}
    for field in table.schema:
        schema[field.name] = str(field.type)
    return schema


def create_backup(db_path):
    """Create a timestamped backup of the LanceDB directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.name}_backup_{timestamp}"
    print(f"Creating backup at {backup_path}")
    if backup_path.exists():
        shutil.rmtree(backup_path)
    shutil.copytree(db_path, backup_path)
    return backup_path


def verify_schema(table):
    """Check if table already has v2 columns."""
    schema = get_table_schema(table)
    v2_columns = {
        "epistemic_status": "string",
        "confidence": "float",
        "entities": "list<item: string>",
        "keywords": "list<item: string>",
        "related_ids": "list<item: string>",
        "is_consolidated": "bool",
        "version": "int32",
    }

    missing = []
    for col, expected_type in v2_columns.items():
        if col not in schema:
            missing.append((col, expected_type))
        else:
            actual_type = schema[col]
            if expected_type not in actual_type:
                print(
                    f"Warning: Column {col} has type {actual_type}, expected {expected_type}"
                )

    if not missing:
        print("✓ Table already has all v2 columns")
        return True
    else:
        print(f"Missing columns: {[c for c, _ in missing]}")
        return False


def migrate_table(table):
    """Migrate table to v2 schema."""
    schema = get_table_schema(table)
    df = table.to_pandas()
    total_rows = len(df)
    print(f"Migrating {total_rows} rows...")

    # Add missing columns with default values
    if "epistemic_status" not in schema:
        df["epistemic_status"] = EPISTEMIC_STATUS_DEFAULT
        print("  Added epistemic_status column")

    if "confidence" not in schema:
        df["confidence"] = CONFIDENCE_DEFAULT
        print("  Added confidence column")

    if "entities" not in schema:
        df["entities"] = [ENTITIES_DEFAULT.copy() for _ in range(total_rows)]
        print("  Added entities column")

    if "keywords" not in schema:
        df["keywords"] = [KEYWORDS_DEFAULT.copy() for _ in range(total_rows)]
        print("  Added keywords column")

    if "related_ids" not in schema:
        df["related_ids"] = [RELATED_IDS_DEFAULT.copy() for _ in range(total_rows)]
        print("  Added related_ids column")

    if "is_consolidated" not in schema:
        df["is_consolidated"] = IS_CONSOLIDATED_DEFAULT
        print("  Added is_consolidated column")

    if "version" not in schema:
        df["version"] = VERSION_DEFAULT
        print("  Added version column")

    # Update version to 2 for all rows
    df["version"] = 2
    print("  Set version=2 for all rows")

    # Write back to table
    # LanceDB doesn't support in-place schema evolution easily, so we replace
    # First create a temporary table with new schema
    db = table._db
    temp_table_name = f"{TABLE_NAME}_migrated_{int(time.time())}"

    # Convert pandas DataFrame to Arrow table
    arrow_table = pa.Table.from_pandas(df)

    # Create new table with updated schema
    new_table = db.create_table(temp_table_name, arrow_table)
    print(f"  Created temporary table {temp_table_name}")

    # Drop old table and rename new one
    db.drop_table(TABLE_NAME)
    db.rename_table(temp_table_name, TABLE_NAME)
    print("  Replaced old table with migrated version")

    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate vector memory schema v1→v2")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify current schema, don't migrate",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        help="Custom backup directory (default: auto-generated)",
    )
    args = parser.parse_args()

    if not VECTOR_MEMORY_PATH.exists():
        print(f"Error: Vector memory directory not found at {VECTOR_MEMORY_PATH}")
        print("Make sure LanceDB vector store exists.")
        sys.exit(1)

    print(f"Vector memory path: {VECTOR_MEMORY_PATH}")

    try:
        db = lancedb.connect(VECTOR_MEMORY_PATH)
        tables = db.table_names()
        if TABLE_NAME not in tables:
            print(f"Error: Table '{TABLE_NAME}' not found in database.")
            print(f"Available tables: {tables}")
            sys.exit(1)

        table = db.open_table(TABLE_NAME)
        print(f"Table '{TABLE_NAME}' opened ({table.count_rows()} rows)")

        # Verify schema
        is_v2 = verify_schema(table)

        if args.verify_only:
            sys.exit(0 if is_v2 else 1)

        if is_v2:
            print("\nSchema already at v2. Migration not needed.")
            # Optional: check if any rows have empty entities/keywords
            df = table.to_pandas(limit=10)
            empty_entities = df["entities"].apply(lambda x: len(x) == 0).sum()
            empty_keywords = df["keywords"].apply(lambda x: len(x) == 0).sum()
            print(f"Rows with empty entities: {empty_entities}/{len(df)}")
            print(f"Rows with empty keywords: {empty_keywords}/{len(df)}")
            if empty_entities == len(df) or empty_keywords == len(df):
                print(
                    "Note: Consider running entity extraction and keyword extraction tools."
                )
            sys.exit(0)

        print("\nSchema needs migration.")
        confirm = input("Proceed with migration? (yes/no): ").strip().lower()
        if confirm not in ("yes", "y"):
            print("Migration cancelled.")
            sys.exit(0)

        # Create backup
        backup_path = create_backup(VECTOR_MEMORY_PATH)
        print(f"Backup created at {backup_path}")

        # Perform migration
        if migrate_table(table):
            print("\n✓ Migration completed successfully")
            print(f"Backup preserved at {backup_path}")
            print("You can restore from backup by copying the backup directory back.")
        else:
            print("\n✗ Migration failed")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
