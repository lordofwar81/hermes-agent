#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")
import lancedb
from pathlib import Path

db_path = Path.home() / ".hermes" / "vector_memory"
print(f"DB path: {db_path}")
db = lancedb.connect(str(db_path))
tables = db.list_tables()
print(f"Tables: {tables}")
if "memory_vectors" in tables:
    table = db.open_table("memory_vectors")
    print(f"Table schema: {table.schema}")
    # Show first few rows
    try:
        df = table.to_pandas()
        print(f"Row count: {len(df)}")
        if len(df) > 0:
            print("First row columns:", df.columns.tolist())
            for col in df.columns:
                print(f"  {col}: {type(df[col].iloc[0])}")
                if col == "vector":
                    val = df[col].iloc[0]
                    if hasattr(val, "shape"):
                        print(f"    shape: {val.shape}, dtype: {val.dtype}")
                    else:
                        print(
                            f"    type: {type(val)}, len: {len(val) if hasattr(val, '__len__') else 'N/A'}"
                        )
            # Show a few rows
            print("\nSample rows:")
            for i, row in df.head(3).iterrows():
                print(
                    f"Row {i}: id={row.get('id', 'N/A')}, text={str(row.get('text', ''))[:80]}..."
                )
    except Exception as e:
        print(f"Error reading table: {e}")
        import traceback

        traceback.print_exc()
