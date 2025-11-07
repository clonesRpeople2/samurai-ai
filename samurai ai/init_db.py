import sqlite3
from datetime import datetime

print("🔧 Ninja Study Planner - Database Cleanup Tool")
print("=" * 60)

conn = sqlite3.connect('database.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Step 1: Check current state
print("\n📊 Current Database State:")
cursor.execute("SELECT COUNT(*) as count FROM users")
user_count = cursor.fetchone()['count']
print(f"   Total users: {user_count}")

cursor.execute("SELECT id, name, email FROM users")
users = cursor.fetchall()
print("\n   User Details:")
for user in users:
    print(f"   - ID: {user['id']}, Name: {user['name']}, Email: {user['email']}")

# Step 2: Find and fix duplicate emails
print("\n🔍 Checking for duplicate emails...")
cursor.execute("""
    SELECT email, COUNT(*) as count, GROUP_CONCAT(id) as ids
    FROM users 
    GROUP BY email 
    HAVING count > 1
""")
duplicates = cursor.fetchall()

if duplicates:
    print(f"   ⚠️  Found {len(duplicates)} duplicate email(s):")
    for dup in duplicates:
        print(f"   - Email: {dup['email']}, IDs: {dup['ids']}")
        
    fix = input("\n   Fix duplicates? (y/n): ").lower()
    if fix == 'y':
        for dup in duplicates:
            ids = [int(x) for x in dup['ids'].split(',')]
            keep_id = min(ids)  # Keep the oldest account
            remove_ids = [x for x in ids if x != keep_id]
            
            print(f"\n   Processing {dup['email']}:")
            print(f"   - Keeping ID: {keep_id}")
            print(f"   - Merging/Removing IDs: {remove_ids}")
            
            # Merge data from duplicate accounts
            for remove_id in remove_ids:
                # Update classes to point to kept user
                cursor.execute(
                    "UPDATE classes SET user_id = ? WHERE user_id = ?",
                    (keep_id, remove_id)
                )
                
                # Update assignments to point to kept user
                cursor.execute(
                    "UPDATE assignments SET user_id = ? WHERE user_id = ?",
                    (keep_id, remove_id)
                )
                
                # Update study plan
                cursor.execute(
                    "UPDATE study_plan SET user_id = ? WHERE user_id = ?",
                    (keep_id, remove_id)
                )
                
                # Delete duplicate user
                cursor.execute("DELETE FROM users WHERE id = ?", (remove_id,))
            
            conn.commit()
            print(f"   ✅ Merged and cleaned up {dup['email']}")
else:
    print("   ✅ No duplicates found")

# Step 3: Check for orphaned data
print("\n🔍 Checking for orphaned data...")
cursor.execute("""
    SELECT COUNT(*) as count FROM classes 
    WHERE user_id NOT IN (SELECT id FROM users)
""")
orphaned_classes = cursor.fetchone()['count']

cursor.execute("""
    SELECT COUNT(*) as count FROM assignments 
    WHERE user_id NOT IN (SELECT id FROM users)
""")
orphaned_assignments = cursor.fetchone()['count']

if orphaned_classes > 0 or orphaned_assignments > 0:
    print(f"   ⚠️  Found orphaned data:")
    print(f"   - Orphaned classes: {orphaned_classes}")
    print(f"   - Orphaned assignments: {orphaned_assignments}")
    
    cleanup = input("\n   Clean up orphaned data? (y/n): ").lower()
    if cleanup == 'y':
        cursor.execute("""
            DELETE FROM classes 
            WHERE user_id NOT IN (SELECT id FROM users)
        """)
        cursor.execute("""
            DELETE FROM assignments 
            WHERE user_id NOT IN (SELECT id FROM users)
        """)
        cursor.execute("""
            DELETE FROM study_plan 
            WHERE user_id NOT IN (SELECT id FROM users)
        """)
        conn.commit()
        print("   ✅ Cleaned up orphaned data")
else:
    print("   ✅ No orphaned data found")

# Step 4: Verify table structure
print("\n🔍 Verifying table structure...")
required_columns = {
    'users': ['id', 'name', 'email', 'xp', 'level', 'streak'],
    'classes': ['id', 'name', 'user_id'],
    'assignments': ['id', 'user_id', 'class_id', 'title', 'due_date', 'completed', 'time_spent_minutes', 'started_at'],
    'study_plan': ['id', 'user_id', 'title', 'start', 'end']
}

all_good = True
for table, columns in required_columns.items():
    cursor.execute(f"PRAGMA table_info({table})")
    existing_cols = [col[1] for col in cursor.fetchall()]
    
    missing = [col for col in columns if col not in existing_cols]
    if missing:
        print(f"   ⚠️  Table '{table}' missing columns: {missing}")
        all_good = False

if all_good:
    print("   ✅ All table structures are correct")

# Step 5: Final summary
print("\n" + "=" * 60)
print("📊 Final Database State:")
cursor.execute("SELECT COUNT(*) as count FROM users")
final_user_count = cursor.fetchone()['count']
print(f"   Total users: {final_user_count}")

cursor.execute("SELECT COUNT(*) as count FROM classes")
class_count = cursor.fetchone()['count']
print(f"   Total classes: {class_count}")

cursor.execute("SELECT COUNT(*) as count FROM assignments")
assignment_count = cursor.fetchone()['count']
print(f"   Total assignments: {assignment_count}")

print("\n✅ Database cleanup complete!")
print("\nNext steps:")
print("1. Replace your auth code in app.py with the cleaned version")
print("2. Restart your Flask app")
print("3. Try logging in with your old account")

conn.close()