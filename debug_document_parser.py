#!/usr/bin/env python3
"""
Debug Script for Document Parser

This script helps debug the document parser with various scenarios.
Set breakpoints in the parser code and run this script to trace execution.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.ingestion.document_parser import (
    parse_document, 
    DocumentParserFactory,
    PDFParser,
    DOCXParser, 
    TXTParser
)


def test_parser_factory():
    """Test the parser factory with different extensions"""
    print("🔧 Testing Parser Factory...")
    
    extensions = ['.pdf', '.docx', '.txt', '.md', '.invalid']
    
    for ext in extensions:
        try:
            parser = DocumentParserFactory.get_parser(ext)
            print(f"✅ {ext}: {type(parser).__name__}")
        except ValueError as e:
            print(f"❌ {ext}: {e}")


def test_file_extension_logic():
    """Test the file extension detection logic"""
    print("\n📄 Testing File Extension Logic...")
    
    test_cases = [
        # (file_path_or_bytes, file_extension, expected_behavior)
        ("test.pdf", None, "Should extract .pdf from path"),
        ("document.docx", None, "Should extract .docx from path"),
        (b"binary content", ".txt", "Should use provided extension"),
        (b"binary content", None, "Should raise ValueError"),
        (Path("example.md"), None, "Should extract .md from Path object"),
    ]
    
    for i, (file_input, extension, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        print(f"Input: {type(file_input).__name__} = {file_input}")
        print(f"Extension: {extension}")
        
        try:
            # This is the line that might trigger "unreachable code" warning
            # Set a breakpoint here to debug the flow
            result = parse_document(file_input, extension)
            print(f"✅ Would succeed with parser for extension")
        except ValueError as e:
            print(f"❌ ValueError: {e}")
        except Exception as e:
            print(f"⚠️ Other error: {e}")


def test_individual_parsers():
    """Test individual parsers with sample content"""
    print("\n🧪 Testing Individual Parsers...")
    
    # Sample text content for testing
    sample_text = "This is a test document.\nIt has multiple lines.\nUsed for debugging."
    
    # Test TXT Parser (easiest to debug)
    print("\n📝 Testing TXT Parser...")
    txt_parser = TXTParser()
    
    try:
        # Test with string content
        result = txt_parser.parse(sample_text.encode('utf-8'))
        print(f"✅ TXT Parser Result: {result['metadata']}")
    except Exception as e:
        print(f"❌ TXT Parser Error: {e}")


def debug_parse_document_function():
    """Debug the main parse_document function step by step"""
    print("\n🐛 Debugging parse_document function...")
    
    # Test case that might cause "unreachable code" confusion
    file_path = "example.txt"
    file_extension = None
    
    print(f"Input: file_path_or_bytes = '{file_path}'")
    print(f"Input: file_extension = {file_extension}")
    
    # Simulate the logic from parse_document function
    print("\n🔍 Checking conditions:")
    
    # Condition 1: file_extension is None and isinstance(file_path_or_bytes, (str, Path))
    condition1 = file_extension is None and isinstance(file_path, (str, Path))
    print(f"Condition 1 (line 139): file_extension is None AND isinstance(file_path, (str, Path)) = {condition1}")
    
    if condition1:
        print("✅ Executing line 140: file_extension = Path(file_path_or_bytes).suffix")
        file_extension = Path(file_path).suffix
        print(f"   Result: file_extension = '{file_extension}'")
    
    # Condition 2: file_extension is None (after potential update from condition 1)
    condition2 = file_extension is None
    print(f"Condition 2 (line 141): file_extension is None = {condition2}")
    
    if condition2:
        print("❌ Would raise ValueError: file_extension must be provided when using bytes input")
    else:
        print("✅ Would proceed to get parser and parse document")


def main():
    """Main debug function"""
    print("🚀 Document Parser Debug Session")
    print("=" * 50)
    
    # Set breakpoints on any of these function calls to debug specific parts
    test_parser_factory()
    test_file_extension_logic()
    test_individual_parsers()
    debug_parse_document_function()
    
    print("\n✨ Debug session complete!")
    print("\n💡 Debugging Tips:")
    print("1. Set breakpoints in src/ingestion/document_parser.py around line 140")
    print("2. Run this script with debugger to trace execution flow")
    print("3. Use 'Debug Document Parser' configuration in VS Code")
    print("4. Check the call stack to understand which condition executes")


if __name__ == "__main__":
    main() 