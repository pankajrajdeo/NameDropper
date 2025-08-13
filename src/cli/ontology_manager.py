#!/usr/bin/env python3
"""
Ontology Database Manager

Comprehensive management tool for ontology database operations:
- Load ontologies from OWL files
- Remove/unload ontologies from database
- Update existing ontologies
- Database maintenance and statistics
- Backup and restore operations
"""

import os
import sys
import argparse
import json
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.ontology_mapper import UnifiedOntologyMapper

def load_ontology(args):
    """Load one or more ontologies into the database."""
    print(f"🧬 Loading Ontology: {args.ontology}")
    print("=" * 60)
    
    with UnifiedOntologyMapper() as mapper:
        # Check if ontology is configured
        if args.ontology not in mapper.ontology_configs:
            print(f"❌ Error: Unknown ontology '{args.ontology}'")
            print(f"Available ontologies: {', '.join(mapper.ontology_configs.keys())}")
            return False
        
        config = mapper.ontology_configs[args.ontology]
        owl_file = config['file']
        ontology_name = config['name']
        
        print(f"📖 Ontology: {ontology_name}")
        print(f"📁 OWL file: {owl_file}")
        
        # Check if OWL file exists
        if not os.path.exists(owl_file):
            print(f"❌ Error: OWL file not found: {owl_file}")
            return False
        
        # Get file size
        file_size = os.path.getsize(owl_file)
        print(f"📊 File size: {file_size / (1024*1024):.1f} MB")
        
        # Check current status
        stats = mapper.get_stats()
        existing_count = stats['ontology_counts'].get(args.ontology, 0)
        
        if existing_count > 0 and not args.force:
            print(f"⚠️  {args.ontology} already has {existing_count} terms in database")
            choice = input(f"Do you want to reload {args.ontology}? (y/N): ").strip().lower()
            if choice not in ['y', 'yes']:
                print(f"❌ Skipping {args.ontology} loading")
                return False
            args.force = True
        
        # Load the ontology
        try:
            print(f"\n🚀 Loading {args.ontology} ontology...")
            mapper.load_ontology_to_database(args.ontology, force_reload=args.force)
            
            # Show results
            final_stats = mapper.get_stats()
            print(f"\n✅ Successfully loaded {args.ontology}!")
            print(f"📊 {args.ontology} terms: {final_stats['ontology_counts'].get(args.ontology, 0):,}")
            print(f"📊 Total terms: {final_stats['total_terms']:,}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading {args.ontology}: {e}")
            return False

def load_all_ontologies(args):
    """Load all configured ontologies."""
    print("🧬 Loading All Ontologies")
    print("=" * 60)
    
    with UnifiedOntologyMapper() as mapper:
        try:
            mapper.load_all_ontologies(force_reload=args.force)
            print("✅ All ontologies loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading ontologies: {e}")
            return False

def remove_ontology(args):
    """Remove an ontology from the database."""
    print(f"🗑️  Removing Ontology: {args.ontology}")
    print("=" * 60)
    
    with UnifiedOntologyMapper() as mapper:
        stats = mapper.get_stats()
        existing_count = stats['ontology_counts'].get(args.ontology, 0)
        
        if existing_count == 0:
            print(f"❌ Ontology '{args.ontology}' not found in database")
            return False
        
        print(f"⚠️  This will remove {existing_count:,} terms from {args.ontology}")
        
        if not args.force:
            choice = input(f"Are you sure you want to remove {args.ontology}? (y/N): ").strip().lower()
            if choice not in ['y', 'yes']:
                print("❌ Operation cancelled")
                return False
        
        try:
            with mapper.conn.cursor() as cur:
                cur.execute("DELETE FROM ontology_terms WHERE ontology = %s", (args.ontology,))
                deleted_count = cur.rowcount
                mapper.conn.commit()
            
            print(f"✅ Removed {deleted_count:,} terms from {args.ontology}")
            
            # Show updated stats
            final_stats = mapper.get_stats()
            print(f"📊 Total terms remaining: {final_stats['total_terms']:,}")
            return True
            
        except Exception as e:
            print(f"❌ Error removing {args.ontology}: {e}")
            return False

def show_stats(args):
    """Show comprehensive database statistics."""
    print("📊 Database Statistics")
    print("=" * 60)
    
    with UnifiedOntologyMapper() as mapper:
        stats = mapper.get_stats()
        
        print(f"🗃️  Total Terms: {stats['total_terms']:,}")
        print(f"🧠 Terms with Embeddings: {stats['terms_with_embeddings']:,}")
        print(f"💾 Database Size: {stats['database_size']}")
        
        print(f"\n📚 Ontologies:")
        for ontology, count in sorted(stats['ontology_counts'].items()):
            config = mapper.ontology_configs.get(ontology, {})
            name = config.get('name', 'Unknown')
            print(f"  {ontology:12} {count:>8,} terms - {name}")
        
        # Check embedding service
        print(f"\n🔌 Embedding Service:")
        if mapper.embedding_service_available:
            print(f"  ✅ Available at {mapper.embedding_service_url}")
        else:
            print(f"  ❌ Not available at {mapper.embedding_service_url}")
        
        # Check database health
        print(f"\n🏥 Database Health:")
        with mapper.conn.cursor() as cur:
            # Check for orphaned records
            cur.execute("SELECT COUNT(*) as count FROM ontology_terms WHERE label IS NULL OR label = ''")
            orphaned = cur.fetchone()['count']
            
            # Check index usage
            cur.execute("""
                SELECT schemaname, relname as tablename, indexrelname as indexname, 
                       idx_tup_read, idx_tup_fetch
                FROM pg_stat_user_indexes 
                WHERE relname = 'ontology_terms' 
                ORDER BY idx_tup_read DESC
                LIMIT 3
            """)
            top_indices = cur.fetchall()
            
            print(f"  Records with missing labels: {orphaned}")
            if top_indices:
                print(f"  Most used indices:")
                for idx in top_indices:
                    print(f"    {idx['indexname']}: {idx['idx_tup_read']:,} reads")

def backup_database(args):
    """Export database to JSON backup."""
    print(f"💾 Creating Database Backup")
    print("=" * 60)
    
    backup_file = args.output or f"ontology_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with UnifiedOntologyMapper() as mapper:
        try:
            print(f"📁 Exporting to: {backup_file}")
            mapper.save_to_json(backup_file)
            
            # Show backup info
            file_size = os.path.getsize(backup_file)
            print(f"✅ Backup created successfully!")
            print(f"📊 File size: {file_size / (1024*1024):.1f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False

def list_ontologies(args):
    """List all available and loaded ontologies."""
    print("📚 Available Ontologies")
    print("=" * 60)
    
    with UnifiedOntologyMapper() as mapper:
        stats = mapper.get_stats()
        
        print("Configured Ontologies:")
        for ont_id, config in sorted(mapper.ontology_configs.items()):
            loaded_count = stats['ontology_counts'].get(ont_id, 0)
            file_exists = os.path.exists(config['file'])
            
            status = "✅ Loaded" if loaded_count > 0 else "⭕ Not loaded"
            file_status = "📁" if file_exists else "❌"
            
            print(f"  {ont_id:12} {status:12} {loaded_count:>8,} terms {file_status} {config['name']}")
            print(f"               File: {config['file']}")
            print()

def rebuild_indexes(args):
    """Rebuild all database indexes for optimal performance."""
    print("🔧 Rebuilding Database Indexes")
    print("=" * 60)
    
    with UnifiedOntologyMapper() as mapper:
        try:
            stats = mapper.get_stats()
            print(f"📊 Database size: {stats['total_terms']:,} terms")
            print(f"⚠️  This may take several minutes for large databases...")
            
            print("\n🏗️  Rebuilding performance-critical indexes...")
            # Call the new public method, which handles its own transactions and vacuuming.
            mapper.rebuild_all_indexes()
            
            print("\n✅ Index rebuild completed!")
            print("💡 Tip: Run this command after loading new ontologies for best performance")
            return True
            
        except Exception as e:
            print(f"❌ Error rebuilding indexes: {e}")
            return False

def vacuum_database(args):
    """Perform database maintenance."""
    print("🧹 Database Maintenance")
    print("=" * 60)
    
    with UnifiedOntologyMapper() as mapper:
        try:
            print("🔄 Running VACUUM ANALYZE...")
            with mapper.conn.cursor() as cur:
                # Can't run VACUUM in a transaction
                mapper.conn.autocommit = True
                cur.execute("VACUUM ANALYZE ontology_terms")
                mapper.conn.autocommit = False
            
            print("🔄 Updating statistics...")
            with mapper.conn.cursor() as cur:
                cur.execute("ANALYZE ontology_terms")
            
            print("✅ Database maintenance completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error during maintenance: {e}")
            return False

def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description='Ontology Database Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s load MMUSDV                    # Load MMUSDV ontology
  %(prog)s load PATO --force              # Force reload PATO
  %(prog)s load-all                       # Load all ontologies
  %(prog)s remove MONDO --force           # Remove MONDO ontology
  %(prog)s stats                          # Show database statistics
  %(prog)s list                           # List all ontologies
  %(prog)s backup --output backup.json    # Create backup
  %(prog)s vacuum                         # Database maintenance
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Load ontology
    load_parser = subparsers.add_parser('load', help='Load an ontology')
    load_parser.add_argument('ontology', help='Ontology to load (e.g., MMUSDV, PATO)')
    load_parser.add_argument('--force', action='store_true', help='Force reload if exists')
    load_parser.add_argument('--no-test', action='store_true', help='Skip search test')
    
    # Load all ontologies
    load_all_parser = subparsers.add_parser('load-all', help='Load all ontologies')
    load_all_parser.add_argument('--force', action='store_true', help='Force reload all')
    
    # Remove ontology
    remove_parser = subparsers.add_parser('remove', help='Remove an ontology')
    remove_parser.add_argument('ontology', help='Ontology to remove')
    remove_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    
    # Statistics
    subparsers.add_parser('stats', help='Show database statistics')
    
    # List ontologies
    subparsers.add_parser('list', help='List all ontologies')
    
    # Backup
    backup_parser = subparsers.add_parser('backup', help='Create database backup')
    backup_parser.add_argument('--output', '-o', help='Output file name')
    
    # Maintenance
    subparsers.add_parser('vacuum', help='Database maintenance')
    
    # Rebuild indexes
    subparsers.add_parser('rebuild-indexes', help='Rebuild all indexes for optimal performance')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    success = False
    if args.command == 'load':
        success = load_ontology(args)
    elif args.command == 'load-all':
        success = load_all_ontologies(args)
    elif args.command == 'remove':
        success = remove_ontology(args)
    elif args.command == 'stats':
        success = show_stats(args)
    elif args.command == 'list':
        success = list_ontologies(args)
    elif args.command == 'backup':
        success = backup_database(args)
    elif args.command == 'vacuum':
        success = vacuum_database(args)
    elif args.command == 'rebuild-indexes':
        success = rebuild_indexes(args)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 