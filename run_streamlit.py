#!/usr/bin/env python3
"""
Cross-platform Streamlit App Launcher

This script ensures the Streamlit app runs correctly across different platforms
and provides helpful setup validation before launching.
"""

import sys
import os
import subprocess
from pathlib import Path


def setup_environment():
    """Set up the environment for running the Streamlit app"""
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Set environment variable for Streamlit to find our modules
    os.environ['PYTHONPATH'] = str(current_dir)


def check_prerequisites():
    """Check if the basic requirements are met"""
    
    print("🔍 Checking prerequisites...")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"   ✅ Streamlit {streamlit.__version__} found")
    except ImportError:
        print("   ❌ Streamlit not found. Install with: pip install streamlit")
        return False
    
    # Check if streamlit_app.py exists
    app_file = Path("streamlit_app.py")
    if not app_file.exists():
        print("   ❌ streamlit_app.py not found in current directory")
        return False
    
    print("   ✅ streamlit_app.py found")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("   ⚠️  .env file not found")
        print("      Create one with your API keys:")
        print("      OPENAI_API_KEY=sk-your-key-here")
        print("      PINECONE_API_KEY=your-key-here")
        return False
    
    print("   ✅ .env file found")
    return True


def run_validation():
    """Run the setup validation script if available"""
    
    validation_script = Path("test_setup.py")
    if validation_script.exists():
        print("\n🧪 Running setup validation...")
        try:
            result = subprocess.run([sys.executable, str(validation_script)], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("   ✅ Validation passed")
                return True
            else:
                print("   ⚠️  Validation issues detected")
                print("   Run 'python test_setup.py' for detailed information")
                
                # Ask user if they want to continue
                response = input("   Continue anyway? (y/n): ").lower().strip()
                return response == 'y'
                
        except subprocess.TimeoutExpired:
            print("   ⚠️  Validation timed out")
            return True
        except Exception as e:
            print(f"   ⚠️  Validation error: {e}")
            return True
    
    return True


def launch_streamlit():
    """Launch the Streamlit application"""
    
    print("\n🚀 Launching Streamlit app...")
    
    # Streamlit command arguments
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_app.py",
        "--server.address", "localhost",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        # Launch Streamlit
        print("🌐 Starting Streamlit server...")
        print("📱 The app will open in your browser automatically")
        print("🔗 If it doesn't open, go to: http://localhost:8501")
        print("\n💡 To stop the server, press Ctrl+C")
        print("=" * 60)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Streamlit app stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        print("\n🔧 Try running manually:")
        print("   streamlit run streamlit_app.py")


def main():
    """Main function to orchestrate the app launch"""
    
    print("📚 RAG Ingestion Pipeline - Streamlit Launcher")
    print("=" * 60)
    
    # Set up environment
    setup_environment()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return False
    
    # Run validation (optional)
    if not run_validation():
        print("\n❌ Validation failed. Please check your setup.")
        return False
    
    # Launch Streamlit
    launch_streamlit()
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please try running: streamlit run streamlit_app.py") 