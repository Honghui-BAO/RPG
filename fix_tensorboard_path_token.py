#!/usr/bin/env python3
"""
Fix script for TensorBoard path length issues in mhl_token branch.
"""

import os
import yaml

def fix_tensorboard_config():
    """Fix TensorBoard configuration to avoid path length issues."""
    
    # Update default configuration
    default_config_path = "genrec/default.yaml"
    if os.path.exists(default_config_path):
        print(f"Updating {default_config_path}...")
        
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set shorter tensorboard log directory
        config['tensorboard_log_dir'] = './logs'
        config['log_dir'] = './logs'
        
        with open(default_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Updated {default_config_path}")
    
    # Update MHL configuration
    mhl_config_path = "genrec/models/MHL/config.yaml"
    if os.path.exists(mhl_config_path):
        print(f"Updating {mhl_config_path}...")
        
        with open(mhl_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add tensorboard configuration
        config['tensorboard_log_dir'] = './logs'
        config['log_dir'] = './logs'
        
        with open(mhl_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Updated {mhl_config_path}")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    print("✅ Created logs directory")
    
    print("\n🎉 TensorBoard path fix completed for mhl_token branch!")
    print("Now you can run the model without path length issues.")

if __name__ == "__main__":
    fix_tensorboard_config()
