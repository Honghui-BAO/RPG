"""
Debug script to check LLADA forward pass
"""
import torch
from genrec.pipeline import Pipeline

# Create pipeline
config_dict = {
    'model': 'LLADA',
    'category': 'Beauty',
    'n_codebook': 32,
    'diffusion_steps': 32,
    'train_batch_size': 2,  # Small batch for debugging
}

pipeline = Pipeline(
    model_name='LLADA',
    dataset_name='AmazonReviews2014',
    config_dict=config_dict
)

# Get a batch
from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    pipeline.tokenized_datasets['train'],
    batch_size=2,
    shuffle=False,
    collate_fn=pipeline.tokenizer.collate_fn['train']
)

batch = next(iter(train_dataloader))
print("=" * 80)
print("Batch info:")
print(f"input_ids shape: {batch['input_ids'].shape}")
print(f"labels shape: {batch['labels'].shape}")
print(f"labels: {batch['labels']}")
print(f"Unique labels: {torch.unique(batch['labels'])}")

# Check item_id2tokens
print("=" * 80)
print("Checking item_id2tokens:")
print(f"item_id2tokens shape: {pipeline.model.item_id2tokens.shape}")
print(f"Sample item_id2tokens[0]: {pipeline.model.item_id2tokens[0]}")  # padding
print(f"Sample item_id2tokens[1]: {pipeline.model.item_id2tokens[1]}")  # first real item

# Get valid labels
labels_flat = batch['labels'].view(-1)
label_mask = (labels_flat != -100) & (labels_flat > 0)
valid_labels = labels_flat[label_mask]
print("=" * 80)
print(f"Valid labels: {valid_labels}")
print(f"Num valid labels: {len(valid_labels)}")

# Check their codes
if len(valid_labels) > 0:
    target_codes = pipeline.model.item_id2tokens[valid_labels]
    print(f"Target codes shape: {target_codes.shape}")
    print(f"Target codes[0]: {target_codes[0]}")
    print(f"Target codes min: {target_codes.min()}")
    print(f"Target codes max: {target_codes.max()}")
    
    # Check each digit
    for i in range(min(3, pipeline.model.n_pred_head)):  # Check first 3 digits
        digit_codes = target_codes[:, i]
        expected_min = i * pipeline.model.config['codebook_size'] + 1
        expected_max = (i + 1) * pipeline.model.config['codebook_size']
        print(f"Digit {i}: range [{digit_codes.min()}, {digit_codes.max()}], expected [{expected_min}, {expected_max}]")
        if digit_codes.min() < expected_min or digit_codes.max() > expected_max:
            print(f"  ⚠️  WARNING: Out of range!")

print("=" * 80)
print("Testing forward pass...")
try:
    batch = {k: v.to(pipeline.config['device']) for k, v in batch.items()}
    outputs = pipeline.model(batch)
    print(f"✅ Forward pass succeeded! Loss: {outputs.loss.item()}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()


