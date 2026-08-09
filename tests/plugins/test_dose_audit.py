"""Tests for the dose audit trail in peptide_tracker.db.

Locks in the 2026-08-08 fix: SQLite triggers on the doses table that
populate dose_audit_log with action='logged'/'modified'/'deleted' for
every dose lifecycle event.

Uses the REAL peptide_tracker.db but isolates test data with a
TEST-COMPOUND marker so it can be cleaned up without affecting real data.

Run: venv/bin/python -m pytest tests/plugins/test_dose_audit.py -v
"""

import sqlite3
import os
import pytest

DB_PATH = os.path.expanduser("~/.hermes/peptide_tracker.db")
TEST_COMPOUND = "TEST-COMPOUND-REGRESSION"


@pytest.fixture
def db():
    """Provide a database connection, clean up test data after each test.

    Cleanup is compound-based: we delete audit entries whose JSON values
    contain the TEST_COMPOUND marker, and any surviving test doses. This
    handles the case where a dose was already deleted (so SELECT from doses
    can't find its ID) but audit entries remain.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    yield conn
    # Cleanup: remove all test-compound audit entries and doses
    try:
        # Delete audit entries that reference the test compound in their JSON
        conn.execute(
            "DELETE FROM dose_audit_log "
            "WHERE new_values LIKE ? OR old_values LIKE ?",
            (f"%{TEST_COMPOUND}%", f"%{TEST_COMPOUND}%"),
        )
        # Delete any surviving test doses (triggers will add delete audit
        # entries, which we clean in the next line — but since we already
        # nuked all compound-referencing audit rows above, those delete
        # triggers produce entries we catch on the NEXT run. To be fully
        # clean, we delete audit entries again after the dose delete.)
        conn.execute("DELETE FROM doses WHERE compound = ?", (TEST_COMPOUND,))
        conn.execute(
            "DELETE FROM dose_audit_log "
            "WHERE new_values LIKE ? OR old_values LIKE ?",
            (f"%{TEST_COMPOUND}%", f"%{TEST_COMPOUND}%"),
        )
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


class TestDoseAuditInsert:
    """Verify INSERT creates an audit entry with action='logged'."""

    def test_insert_creates_logged_audit(self, db):
        cursor = db.execute(
            "INSERT INTO doses (compound, dose_units, site, timestamp, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (TEST_COMPOUND, "250mcg", "left_glute", "2026-08-08T08:00", "test dose"),
        )
        db.commit()
        dose_id = cursor.lastrowid

        # Verify the dose was inserted
        row = db.execute(
            "SELECT * FROM doses WHERE id = ?", (dose_id,)
        ).fetchone()
        assert row is not None
        assert row["compound"] == TEST_COMPOUND

        # Verify audit entry created
        audit = db.execute(
            "SELECT * FROM dose_audit_log WHERE dose_id = ? AND action = 'logged'",
            (dose_id,),
        ).fetchone()
        assert audit is not None, "INSERT must create 'logged' audit entry"
        assert audit["action"] == "logged"
        assert audit["new_values"] is not None
        # new_values should be JSON containing the compound
        assert TEST_COMPOUND in audit["new_values"]


class TestDoseAuditUpdate:
    """Verify UPDATE creates an audit entry with action='modified'."""

    def test_update_creates_modified_audit(self, db):
        # Insert a dose first
        cursor = db.execute(
            "INSERT INTO doses (compound, dose_units, site, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (TEST_COMPOUND, "250mcg", "left_glute", "2026-08-08T08:00"),
        )
        db.commit()
        dose_id = cursor.lastrowid

        # Clear any audit entries from the insert (isolate the update audit)
        db.execute("DELETE FROM dose_audit_log WHERE dose_id = ?", (dose_id,))
        db.commit()

        # Update the dose
        db.execute(
            "UPDATE doses SET dose_units = ?, notes = ? WHERE id = ?",
            ("500mcg", "updated dose", dose_id),
        )
        db.commit()

        # Verify audit entry
        audit = db.execute(
            "SELECT * FROM dose_audit_log WHERE dose_id = ? AND action = 'modified'",
            (dose_id,),
        ).fetchone()
        assert audit is not None, "UPDATE must create 'modified' audit entry"
        assert audit["action"] == "modified"
        assert audit["old_values"] is not None
        assert audit["new_values"] is not None
        # old_values should have old dose, new_values should have new dose
        assert "250mcg" in audit["old_values"]
        assert "500mcg" in audit["new_values"]


class TestDoseAuditDelete:
    """Verify DELETE creates an audit entry with action='deleted'."""

    def test_delete_creates_deleted_audit(self, db):
        # Insert a dose first
        cursor = db.execute(
            "INSERT INTO doses (compound, dose_units, site, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (TEST_COMPOUND, "250mcg", "left_glute", "2026-08-08T08:00"),
        )
        db.commit()
        dose_id = cursor.lastrowid

        # Clear audit entries from insert
        db.execute("DELETE FROM dose_audit_log WHERE dose_id = ?", (dose_id,))
        db.commit()

        # Delete the dose
        db.execute("DELETE FROM doses WHERE id = ?", (dose_id,))
        db.commit()

        # Verify the dose is gone
        row = db.execute(
            "SELECT * FROM doses WHERE id = ?", (dose_id,)
        ).fetchone()
        assert row is None

        # Verify audit entry
        audit = db.execute(
            "SELECT * FROM dose_audit_log WHERE dose_id = ? AND action = 'deleted'",
            (dose_id,),
        ).fetchone()
        assert audit is not None, "DELETE must create 'deleted' audit entry"
        assert audit["action"] == "deleted"
        assert audit["old_values"] is not None
        assert TEST_COMPOUND in audit["old_values"]


class TestDoseAuditFullLifecycle:
    """Verify the full insert → update → delete lifecycle creates 3 audit entries."""

    def test_full_lifecycle_three_audit_entries(self, db):
        # INSERT
        cursor = db.execute(
            "INSERT INTO doses (compound, dose_units, timestamp) VALUES (?, ?, ?)",
            (TEST_COMPOUND, "250mcg", "2026-08-08T08:00"),
        )
        db.commit()
        dose_id = cursor.lastrowid

        # UPDATE
        db.execute(
            "UPDATE doses SET dose_units = ? WHERE id = ?",
            ("350mcg", dose_id),
        )
        db.commit()

        # DELETE
        db.execute("DELETE FROM doses WHERE id = ?", (dose_id,))
        db.commit()

        # Count audit entries
        audits = db.execute(
            "SELECT * FROM dose_audit_log WHERE dose_id = ? ORDER BY audit_id",
            (dose_id,),
        ).fetchall()

        assert len(audits) == 3, (
            f"Full lifecycle should create 3 audit entries, got {len(audits)}"
        )
        actions = [a["action"] for a in audits]
        assert actions == ["logged", "modified", "deleted"], (
            f"Expected ['logged', 'modified', 'deleted'], got {actions}"
        )


class TestAuditIntegrity:
    """Verify audit entries contain correct JSON snapshots."""

    def test_logged_entry_has_new_values_only(self, db):
        cursor = db.execute(
            "INSERT INTO doses (compound, dose_units, site, timestamp, cycle_day, week) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (TEST_COMPOUND, "250mcg", "right_glute", "2026-08-08T08:00", 5, 1),
        )
        db.commit()
        dose_id = cursor.lastrowid

        audit = db.execute(
            "SELECT * FROM dose_audit_log WHERE dose_id = ? AND action = 'logged'",
            (dose_id,),
        ).fetchone()

        import json
        new_vals = json.loads(audit["new_values"])
        assert new_vals["compound"] == TEST_COMPOUND
        assert new_vals["dose_units"] == "250mcg"
        assert new_vals["site"] == "right_glute"
        assert new_vals["cycle_day"] == 5
        assert new_vals["week"] == 1
        # logged action should have NO old_values
        assert audit["old_values"] is None
