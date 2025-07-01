#!/usr/bin/env python3
"""
Dependency Fix Script for RAG Pipeline

This script helps resolve common dependency conflicts and environment issues.
"""

import subprocess
import sys
import importlib
import os

def run_command(command, description):
    """Run a shell command and report the result"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def uninstall_conflicting_packages():
    """Remove packages that might conflict"""
    print("🗑️  Removing potentially conflicting packages...")
    
    conflicting_packages = [
        "docx",  # This conflicts with python-docx
        "exceptions",  # Sometimes installed accidentally
    ]
    
    for package in conflicting_packages:
        print(f"   Checking {package}...")
        try:
            importlib.import_module(package)
            print(f"   🗑️  Uninstalling {package}...")
            run_command(f"{sys.executable} -m pip uninstall {package} -y", f"Removing {package}")
        except ImportError:
            print(f"   ✅ {package} not installed (good)")

def reinstall_requirements():
    """Reinstall requirements from scratch"""
    print("\n📦 Reinstalling requirements...")
    
    # First, upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    run_command(f"{sys.executable} -m pip install -r requirements.txt --force-reinstall", "Installing requirements")

def verify_critical_imports():
    """Verify that critical packages can be imported"""
    print("\n🔍 Verifying critical imports...")
    
    critical_imports = [
        ("docx", "python-docx for DOCX processing"),
        ("PyPDF2", "PyPDF2 for PDF processing"),
        ("streamlit", "Streamlit for the web interface"),
        ("langchain", "LangChain core"),
        ("pinecone", "Pinecone vector database"),
    ]
    
    failed_imports = []
    
    for module, description in critical_imports:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module} - {description}")
        except ImportError as e:
            print(f"   ❌ {module} - {description} (FAILED: {e})")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def check_python_path():
    """Check and fix Python path issues"""
    print("\n🐍 Checking Python path...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add current directory to Python path if not present
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"   ✅ Added {current_dir} to Python path")
    else:
        print(f"   ✅ Python path is correct")

def main():
    """Main fix function"""
    print("🔧 RAG Pipeline Dependency Fix Script")
    print("=" * 50)
    
    # Step 1: Check Python path
    check_python_path()
    
    # Step 2: Remove conflicting packages
    uninstall_conflicting_packages()
    
    # Step 3: Reinstall requirements
    reinstall_requirements()
    
    # Step 4: Verify imports
    print("\n" + "=" * 50)
    if verify_critical_imports():
        print("🎉 All critical imports working!")
        print("\n✅ Dependencies fixed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Run: python test_setup.py")
        print("   2. Run: streamlit run streamlit_app.py")
    else:
        print("❌ Some imports still failing.")
        print("\n💡 Manual steps to try:")
        print("   1. Create a new virtual environment:")
        print("      python -m venv rag_env")
        print("      source rag_env/bin/activate  # On Windows: rag_env\\Scripts\\activate")
        print("   2. Install requirements:")
        print("      pip install -r requirements.txt")
        print("   3. Run the test again:")
        print("      python test_setup.py")

if __name__ == "__main__":
    main() 