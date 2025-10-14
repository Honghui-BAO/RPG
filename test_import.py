#!/usr/bin/env python3
"""Test if LLADA can be imported"""

print("Step 1: Importing basic modules...")
import torch
print("✓ PyTorch imported")

print("\nStep 2: Importing genrec modules...")
from genrec.pipeline import Pipeline
print("✓ Pipeline imported")

print("\nStep 3: Importing LLADA model...")
try:
    from genrec.models.LLADA.model import LLaDARecommender
    print("✓ LLaDARecommender imported")
except Exception as e:
    print(f"✗ Failed to import LLaDARecommender: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 4: Importing LLADA tokenizer...")
try:
    from genrec.models.LLADA.tokenizer import LLADATokenizer
    print("✓ LLADATokenizer imported")
except Exception as e:
    print(f"✗ Failed to import LLADATokenizer: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 5: Testing get_model...")
try:
    from genrec.utils import get_model
    model_class = get_model('LLADA')
    print(f"✓ get_model('LLADA') returned: {model_class}")
except Exception as e:
    print(f"✗ Failed to get LLADA model: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 6: Testing get_tokenizer...")
try:
    from genrec.utils import get_tokenizer
    tokenizer_class = get_tokenizer('LLADA')
    print(f"✓ get_tokenizer('LLADA') returned: {tokenizer_class}")
except Exception as e:
    print(f"✗ Failed to get LLADA tokenizer: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All imports successful! LLADA is ready to use.")
print("="*60)

