#!/usr/bin/env python3
"""
Memory Export - Backup and export memories.

Formats:
- JSON: Full export with metadata
- SQLite: Database dump
- Markdown: Human-readable

Usage:
    from tools.memory_export import MemoryExporter

    exporter = MemoryExporter(vector_memory, bm25_store)

    # Export JSON
    result = exporter.export(format="json", include_epistemic=True)

    # Create full backup
    backup = exporter.create_backup()
"""

import json
import logging
import math
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from tools.registry import registry

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Memory Exporter
# -----------------------------------------------------------------------------


class MemoryExporter:
    """Export memories to various formats."""

    def __init__(self, vector_store, bm25_store=None):
        """
        Initialize exporter.

        Args:
            vector_store: LanceDB table object
            bm25_store: BM25MemoryStore instance (optional)
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def export(
        self,
        format: str = "json",
        output_path: Optional[Path] = None,
        include_epistemic: bool = True,
        include_vectors: bool = False,
    ) -> Dict[str, Any]:
        """
        Export memories to specified format.

        Args:
            format: "json", "sqlite", or "markdown"
            output_path: Path to output file (default: auto-generated)
            include_epistemic: Include epistemic status and confidence
            include_vectors: Include embedding vectors (large!)

        Returns:
            Dict with export metadata
        """
        if format not in ("json", "sqlite", "markdown"):
            raise ValueError(f"Unsupported format: {format}")

        # Get memories from vector store
        memories = self._get_memories(include_vectors)

        # Apply epistemic filter
        if not include_epistemic:
            for mem in memories:
                mem.pop("epistemic_status", None)
                mem.pop("confidence", None)

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path.cwd() / f"memories_export_{timestamp}.{format}"

        # Export based on format
        if format == "json":
            return self._export_json(memories, output_path)
        elif format == "sqlite":
            return self._export_sqlite(memories, output_path)
        elif format == "markdown":
            return self._export_markdown(memories, output_path)

    def _get_memories(self, include_vectors: bool = False) -> List[Dict[str, Any]]:
        """Retrieve all memories from vector store."""
        if self.vector_store is None:
            return []

        try:
            df = self.vector_store.to_pandas()
            logger.debug(
                f"DataFrame shape: {df.shape if hasattr(df, 'shape') else 'no shape'}, columns: {df.columns.tolist() if df is not None else 'None'}"
            )
            memories = []

            for _, row in df.iterrows():
                # Helper to safely get values
                def safe_get(key, default):
                    val = row.get(key)
                    if val is None:
                        return default
                    if (
                        isinstance(default, float)
                        and isinstance(val, float)
                        and math.isnan(val)
                    ):
                        return default
                    if (
                        isinstance(default, int)
                        and isinstance(val, float)
                        and math.isnan(val)
                    ):
                        return default
                    return val

                mem = {
                    "id": safe_get("id", ""),
                    "text": safe_get("text", ""),
                    "source": safe_get("source", ""),
                    "memory_type": safe_get("memory_type", ""),
                    "session_id": safe_get("session_id", ""),
                    "created_at": safe_get("created_at", 0.0),
                    "access_count": int(safe_get("access_count", 0)),
                    "epistemic_status": safe_get("epistemic_status", "stated"),
                    "confidence": float(safe_get("confidence", 0.5)),
                    "entities": list(safe_get("entities", [])),
                    "keywords": list(safe_get("keywords", [])),
                    "related_ids": list(safe_get("related_ids", [])),
                    "version": int(safe_get("version", 1)),
                }

                if include_vectors and "vector" in row:
                    vec = row["vector"]
                    if hasattr(vec, "tolist"):
                        mem["vector"] = vec.tolist()
                    else:
                        mem["vector"] = list(vec)

                memories.append(mem)

            return memories
        except Exception as e:
            logger.error(f"Failed to get memories: {e}")
            return []

    def _export_json(
        self,
        memories: List[Dict[str, Any]],
        output_path: Path,
    ) -> Dict[str, Any]:
        """Export memories to JSON file."""
        export_data = {
            "export_date": datetime.now().isoformat(),
            "memory_count": len(memories),
            "memories": memories,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "format": "json",
            "output_path": str(output_path),
            "memory_count": len(memories),
            "file_size": output_path.stat().st_size,
        }

    def _export_sqlite(
        self,
        memories: List[Dict[str, Any]],
        output_path: Path,
    ) -> Dict[str, Any]:
        """Export memories to SQLite database."""
        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()

        # Create memories table
        cursor.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT,
                memory_type TEXT,
                session_id TEXT,
                created_at REAL,
                access_count INTEGER,
                epistemic_status TEXT,
                confidence REAL,
                entities TEXT,  -- JSON array
                keywords TEXT,  -- JSON array
                related_ids TEXT,  -- JSON array
                version INTEGER,
                exported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert memories
        for mem in memories:
            cursor.execute(
                """
                INSERT INTO memories (
                    id, text, source, memory_type, session_id,
                    created_at, access_count, epistemic_status, confidence,
                    entities, keywords, related_ids, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    mem["id"],
                    mem["text"],
                    mem["source"],
                    mem["memory_type"],
                    mem["session_id"],
                    mem["created_at"],
                    mem["access_count"],
                    mem["epistemic_status"],
                    mem["confidence"],
                    json.dumps(mem["entities"]),
                    json.dumps(mem["keywords"]),
                    json.dumps(mem["related_ids"]),
                    mem["version"],
                ),
            )

        conn.commit()
        conn.close()

        return {
            "success": True,
            "format": "sqlite",
            "output_path": str(output_path),
            "memory_count": len(memories),
            "file_size": output_path.stat().st_size,
        }

    def _export_markdown(
        self,
        memories: List[Dict[str, Any]],
        output_path: Path,
    ) -> Dict[str, Any]:
        """Export memories to Markdown file."""
        lines = [
            "# Memory Export",
            f"Exported: {datetime.now().isoformat()}",
            f"Total memories: {len(memories)}",
            "",
            "---",
            "",
        ]

        for i, mem in enumerate(memories, 1):
            lines.append(f"## Memory {i}: {mem['id']}")
            lines.append("")
            lines.append(f"**Text:** {mem['text']}")
            lines.append("")
            lines.append(f"**Source:** {mem['source']}")
            lines.append(f"**Type:** {mem['memory_type']}")
            lines.append(
                f"**Epistemic Status:** {mem['epistemic_status']} (confidence: {mem['confidence']:.2f})"
            )
            lines.append(
                f"**Created:** {datetime.fromtimestamp(mem['created_at']).isoformat() if mem['created_at'] else 'unknown'}"
            )
            lines.append(f"**Access Count:** {mem['access_count']}")
            lines.append("")

            if mem["entities"]:
                lines.append(f"**Entities:** {', '.join(mem['entities'])}")

            if mem["keywords"]:
                lines.append(f"**Keywords:** {', '.join(mem['keywords'])}")

            if mem["related_ids"]:
                lines.append(f"**Related IDs:** {', '.join(mem['related_ids'])}")

            lines.append(f"**Version:** {mem['version']}")
            lines.append("")
            lines.append("---")
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return {
            "success": True,
            "format": "markdown",
            "output_path": str(output_path),
            "memory_count": len(memories),
            "file_size": output_path.stat().st_size,
        }

    def create_backup(
        self,
        backup_dir: Optional[Path] = None,
        include_all_formats: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a comprehensive backup of memory stores.

        Args:
            backup_dir: Directory for backup (default: ~/.hermes/backups)
            include_all_formats: Export in JSON, SQLite, and Markdown

        Returns:
            Dict with backup metadata
        """
        if backup_dir is None:
            backup_dir = Path.home() / ".hermes" / "backups"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"memory_backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Export to all formats if requested, otherwise just SQLite (most complete)
        if include_all_formats:
            formats = ["json", "sqlite", "markdown"]
        else:
            formats = ["sqlite"]

        for fmt in formats:
            output_file = backup_path / f"memories.{fmt}"
            result = self.export(
                format=fmt,
                output_path=output_file,
                include_epistemic=True,
                include_vectors=False,
            )
            results[fmt] = result

        # Also backup the LanceDB directory if possible
        try:
            # Determine LanceDB path from vector store
            # This is hacky - depends on lancedb implementation
            import lancedb

            # We'll just copy the entire vector_memory directory
            vector_memory_dir = Path.home() / ".hermes" / "vector_memory"
            if vector_memory_dir.exists():
                backup_vector_dir = backup_path / "vector_memory"
                shutil.copytree(vector_memory_dir, backup_vector_dir)
                results["vector_store"] = {
                    "backup_path": str(backup_vector_dir),
                    "original_path": str(vector_memory_dir),
                }
        except Exception as e:
            logger.warning(f"Failed to backup vector store: {e}")

        return {
            "success": True,
            "backup_path": str(backup_path),
            "timestamp": timestamp,
            "results": results,
        }


# -----------------------------------------------------------------------------
# Tool integration
# -----------------------------------------------------------------------------


def memory_export_tool(args: Dict[str, Any], **kwargs) -> str:
    """
    Tool handler for memory export.

    Expected args:
        format: str (optional) - "json", "sqlite", or "markdown" (default: "json")
        output_path: str (optional) - Path to output file
        include_epistemic: bool (optional) - Include epistemic status (default: true)
        include_vectors: bool (optional) - Include embedding vectors (default: false)
        backup: bool (optional) - Create full backup instead of single export
    """
    try:
        import lancedb
        from .bm25_memory import BM25MemoryStore

        # Get parameters
        format_type = args.get("format", "json")
        output_path_str = args.get("output_path")
        include_epistemic = args.get("include_epistemic", True)
        include_vectors = args.get("include_vectors", False)
        backup = args.get("backup", False)

        # Initialize stores
        db_path = Path.home() / ".hermes" / "vector_memory"
        db = lancedb.connect(str(db_path))

        # Check if table exists
        tables = db.list_tables()
        # Handle both ListTablesResponse object and plain list
        if hasattr(tables, "tables"):
            actual_tables = tables.tables
        else:
            actual_tables = tables
        if "memory_vectors" not in actual_tables:
            return json.dumps(
                {
                    "error": "Vector memory table not found",
                    "suggestion": "Add memories first using memory tool",
                }
            )

        vector_store = db.open_table("memory_vectors")
        bm25_store = BM25MemoryStore()

        # Create exporter
        exporter = MemoryExporter(vector_store, bm25_store)

        if backup:
            # Create full backup
            result = exporter.create_backup(include_all_formats=True)
        else:
            # Single format export
            output_path = Path(output_path_str) if output_path_str else None
            result = exporter.export(
                format=format_type,
                output_path=output_path,
                include_epistemic=include_epistemic,
                include_vectors=include_vectors,
            )

        return json.dumps(result, indent=2)

    except ImportError as e:
        return json.dumps({"error": f"Required module missing: {e}"})
    except Exception as e:
        logger.error(f"Memory export failed: {e}")
        return json.dumps({"error": f"Export failed: {str(e)}"})


def check_memory_export_requirements() -> bool:
    """Check if requirements for memory export are met."""
    try:
        import lancedb

        return True
    except ImportError:
        return False


# Schema for tool registration
MEMORY_EXPORT_SCHEMA = {
    "name": "memory_export",
    "description": "Export memories to JSON, SQLite, or Markdown format. Optionally create full backup.",
    "parameters": {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["json", "sqlite", "markdown"],
                "description": "Export format (default: json)",
                "default": "json",
            },
            "output_path": {
                "type": "string",
                "description": "Path to output file (default: auto-generated)",
            },
            "include_epistemic": {
                "type": "boolean",
                "description": "Include epistemic status and confidence (default: true)",
                "default": True,
            },
            "include_vectors": {
                "type": "boolean",
                "description": "Include embedding vectors (large files!) (default: false)",
                "default": False,
            },
            "backup": {
                "type": "boolean",
                "description": "Create full backup in all formats (default: false)",
                "default": False,
            },
        },
        "required": [],
    },
}

# Register the tool
registry.register(
    name="memory_export",
    toolset="memory",
    schema=MEMORY_EXPORT_SCHEMA,
    handler=memory_export_tool,
    check_fn=check_memory_export_requirements,
    emoji="💾",
)

# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory export test")
    parser.add_argument(
        "--format", choices=["json", "sqlite", "markdown"], default="json"
    )
    parser.add_argument("--output", type=Path, help="Output path")
    parser.add_argument(
        "--no-epistemic", action="store_true", help="Exclude epistemic status"
    )
    parser.add_argument(
        "--include-vectors", action="store_true", help="Include vectors"
    )
    parser.add_argument("--backup", action="store_true", help="Create full backup")

    args = parser.parse_args()

    # Try to initialize vector store
    try:
        import lancedb
        from .bm25_memory import BM25MemoryStore

        db_path = Path.home() / ".hermes" / "vector_memory"
        db = lancedb.connect(db_path)
        vector_store = db.open_table("memory_vectors")

        bm25_store = BM25MemoryStore()

        exporter = MemoryExporter(vector_store, bm25_store)

        if args.backup:
            result = exporter.create_backup(include_all_formats=True)
            print("Backup created:")
            print(json.dumps(result, indent=2))
        else:
            result = exporter.export(
                format=args.format,
                output_path=args.output,
                include_epistemic=not args.no_epistemic,
                include_vectors=args.include_vectors,
            )
            print("Export result:")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
