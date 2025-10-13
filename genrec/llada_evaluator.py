# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Extended evaluator for LLADA model with code validity tracking
"""

from genrec.evaluator import Evaluator


class LLaDAEvaluator(Evaluator):
    """
    Evaluator with additional metrics for LLADA diffusion model
    Tracks code validity statistics
    """
    
    def __init__(self, config: dict, tokenizer):
        super().__init__(config, tokenizer)
        
        # Statistics for code validity
        self.total_generated = 0
        self.total_exact_matches = 0
        self.total_max_matching_codes = []
    
    def update_validity_stats(self, validity_stats: dict):
        """
        Update code validity statistics
        
        Args:
            validity_stats: dict from model.check_code_validity()
        """
        self.total_generated += validity_stats['total']
        self.total_exact_matches += validity_stats['exact_matches']
        self.total_max_matching_codes.append(validity_stats['avg_max_matching_codes'])
    
    def get_validity_summary(self) -> dict:
        """
        Get summary of code validity across all evaluations
        
        Returns:
            dict with validity metrics
        """
        if self.total_generated == 0:
            return {
                'exact_match_rate': 0.0,
                'avg_max_matching_codes': 0.0
            }
        
        return {
            'exact_match_rate': self.total_exact_matches / self.total_generated,
            'avg_max_matching_codes': sum(self.total_max_matching_codes) / len(self.total_max_matching_codes)
        }
    
    def reset_validity_stats(self):
        """Reset validity statistics"""
        self.total_generated = 0
        self.total_exact_matches = 0
        self.total_max_matching_codes = []

