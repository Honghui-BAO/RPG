#!/usr/bin/env python3
"""
Test script to verify MHL model can be imported and initialized properly.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mhl_import():
    """Test if MHL model can be imported successfully."""
    try:
        from genrec.models import MHL
        print("✓ MHL model imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import MHL model: {e}")
        return False

def test_mhl_tokenizer_import():
    """Test if MHL tokenizer can be imported successfully."""
    try:
        from genrec.models.MHL.tokenizer import MHLTokenizer
        print("✓ MHL tokenizer imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import MHL tokenizer: {e}")
        return False

def test_model_loading():
    """Test if MHL model can be loaded through the utils system."""
    try:
        from genrec.utils import get_model, get_tokenizer
        
        # Test model loading
        model_class = get_model('MHL')
        print("✓ MHL model class loaded through utils")
        
        # Test tokenizer loading
        tokenizer_class = get_tokenizer('MHL')
        print("✓ MHL tokenizer class loaded through utils")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load MHL through utils: {e}")
        return False

if __name__ == "__main__":
    print("Testing MHL model setup...")
    print("=" * 50)
    
    success = True
    success &= test_mhl_import()
    success &= test_mhl_tokenizer_import()
    success &= test_model_loading()
    
    print("=" * 50)
    if success:
        print("✓ All tests passed! MHL model is ready to use.")
        print("\nTo run MHL model, use:")
        print("python main.py --model=MHL --category=Beauty")
    else:
        print("✗ Some tests failed. Please check the errors above.")
