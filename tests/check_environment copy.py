#!/usr/bin/env python3
"""
Environment check and setup validation for Anigma F1 AI Agent
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking Environment Setup...")
    print("=" * 50)
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("❌ Python 3.8+ required")
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check dependencies
    required_packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"), 
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence_transformers"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("requests", "requests"),
        ("python-dotenv", "dotenv")
    ]
    
    missing_packages = []
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {display_name}")
        except ImportError:
            missing_packages.append(display_name)
            print(f"❌ {display_name}")
    
    if missing_packages:
        issues.append(f"Missing packages: {', '.join(missing_packages)}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - using CPU (slower)")
    except ImportError:
        pass
    
    # Check HF token
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("✅ HF_TOKEN configured")
    else:
        print("⚠️  HF_TOKEN not set - remote model unavailable")
    
    # Check file structure
    expected_files = [
        "src/agent/agent.py",
        "src/agent/model_loader.py", 
        "src/agent/remote_qwen_tool.py",
        "src/infra/vector_store.py",
        "src/presentation/cli_client.py",
        "main.py"
    ]
    
    for file_path in expected_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            issues.append(f"Missing file: {file_path}")
            print(f"❌ {file_path}")
    
    # Summary
    print("\n" + "=" * 50)
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Environment check passed!")
        return True

def fix_common_issues():
    """Suggest fixes for common issues"""
    print("\n🔧 Common Fixes:")
    print("- Install dependencies: uv sync")
    print("- Set HF token: export HF_TOKEN=your_token")
    print("- CUDA setup: Install CUDA toolkit if needed")
    print("- WSL2 GPU: Ensure WSL2 has GPU access configured")

if __name__ == "__main__":
    print("🚀 Anigma F1 AI Agent - Environment Check")
    success = check_environment()
    if not success:
        fix_common_issues()
    sys.exit(0 if success else 1)