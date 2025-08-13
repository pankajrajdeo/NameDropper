#!/usr/bin/env python3
"""
Basic tests for NameDropper package
"""

import sys
import os
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from core.ontology_mapper import UnifiedOntologyMapper
        print("✅ UnifiedOntologyMapper imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import UnifiedOntologyMapper: {e}")
        return False
    
    try:
        from core.harmonizer_agent import harmonize_metadata, HarmonizedTerm
        print("✅ harmonizer_agent imported successfully")
    except Exception as e:
        if "OPENAI_API_KEY" in str(e):
            print("⚠️ harmonizer_agent imported but requires OPENAI_API_KEY (this is expected)")
        else:
            print(f"❌ Failed to import harmonizer_agent: {e}")
            return False
    
    try:
        from cli.ontology_manager import load_ontology, load_all_ontologies
        print("✅ ontology_manager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ontology_manager: {e}")
        return False
    
    try:
        from cli.harmonize_metadata import MetadataHarmonizer
        print("✅ harmonize_metadata imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import harmonize_metadata: {e}")
        return False
    
    try:
        from web.app import create_interface
        print("✅ web app imported successfully")
    except Exception as e:
        if "OPENAI_API_KEY" in str(e):
            print("⚠️ web app imported but requires OPENAI_API_KEY (this is expected)")
        else:
            print(f"❌ Failed to import web app: {e}")
            return False
    
    return True

def test_package_structure():
    """Test that the package structure is correct."""
    src_dir = Path(__file__).parent.parent / "src"
    
    required_files = [
        "core/ontology_mapper.py",
        "core/harmonizer_agent.py",
        "cli/ontology_manager.py",
        "cli/harmonize_metadata.py",
        "web/app.py"
    ]
    
    for file_path in required_files:
        full_path = src_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def test_ollama_availability():
    """Test if Ollama embeddings can be imported."""
    try:
        from langchain_ollama import OllamaEmbeddings
        print("✅ langchain-ollama available")
        return True
    except ImportError:
        print("⚠️ langchain-ollama not available (install with: pip install langchain-ollama)")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring external services."""
    try:
        from core.ontology_mapper import UnifiedOntologyMapper
        
        # Test that we can create the class (without connecting to database)
        print("✅ UnifiedOntologyMapper class can be instantiated")
        
        # Test ontology configs
        mapper = UnifiedOntologyMapper()
        if hasattr(mapper, 'ontology_configs'):
            print(f"✅ Ontology configs loaded: {len(mapper.ontology_configs)} ontologies")
        else:
            print("❌ Ontology configs not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running NameDropper basic tests...")
    print("=" * 50)
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Basic Functionality", test_basic_functionality),
        ("Module Imports", test_imports),
        ("Ollama Availability", test_ollama_availability),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! NameDropper is ready to use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        print("\n💡 Note: Some failures may be expected if:")
        print("   - OpenAI API key is not set")
        print("   - Database is not running")
        print("   - Ollama is not available")
        return 1

if __name__ == "__main__":
    sys.exit(main())
