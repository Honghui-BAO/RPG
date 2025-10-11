# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Tokenizer for LLaDA model - extends RPG tokenizer with MASK token

Note: This tokenizer reuses all semantic IDs from RPG (no need to retrain OPQ).
Only adds a MASK token for diffusion training.
"""

from genrec.models.RPG.tokenizer import RPGTokenizer


class LLaDATokenizer(RPGTokenizer):
    """
    LLaDA Tokenizer with MASK token support
    
    Inherits from RPGTokenizer to reuse existing semantic IDs.
    The only change is adding a MASK token for diffusion.
    
    Vocabulary structure:
        0: [PAD]
        1-256: digit 0 codes
        257-512: digit 1 codes
        ...
        7937-8192: digit 31 codes
        8193: [EOS]
        8194: [MASK]  ← New for LLaDA
    """
    
    def __init__(self, config: dict, dataset):
        # Inherit all RPG tokenization (semantic IDs, OPQ, etc.)
        super().__init__(config, dataset)
        
        # Add MASK token after EOS
        self.mask_token_id = self.eos_token + 1
        
        self.log(f'[TOKENIZER] Added MASK token at position {self.mask_token_id}')

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size including MASK token"""
        return self.mask_token_id + 1  # 8195 = 8193 (EOS) + 1 (MASK) + 1

