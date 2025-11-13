from load_data import T5Dataset, load_lines
import os

def compute_statistics_before(data_folder, split):
    """Compute statistics BEFORE preprocessing (raw text)"""
    # Load raw text files
    nl_queries = load_lines(os.path.join(data_folder, f'{split}.nl'))
    sql_queries = load_lines(os.path.join(data_folder, f'{split}.sql'))
    
    # Mean sentence length (words)
    nl_lengths = [len(query.split()) for query in nl_queries]
    mean_nl_length = sum(nl_lengths) / len(nl_lengths)
    
    # Mean SQL query length (words)
    sql_lengths = [len(query.split()) for query in sql_queries]
    mean_sql_length = sum(sql_lengths) / len(sql_lengths)
    
    # Vocabulary size (natural language)
    nl_vocab = set()
    for query in nl_queries:
        nl_vocab.update(query.lower().split())
    
    # Vocabulary size (SQL)
    sql_vocab = set()
    for query in sql_queries:
        sql_vocab.update(query.split())
    
    return {
        'num_examples': len(nl_queries),
        'mean_nl_length': mean_nl_length,
        'mean_sql_length': mean_sql_length,
        'vocab_size_nl': len(nl_vocab),
        'vocab_size_sql': len(sql_vocab)
    }

def compute_statistics_after(dataset):
    """Compute statistics AFTER preprocessing (tokenized)"""
    # Mean encoder token length
    enc_lengths = [len(tokens) for tokens in dataset.encoder_inputs]
    mean_enc_length = sum(enc_lengths) / len(enc_lengths)
    
    # Mean decoder token length (if available)
    if hasattr(dataset, 'decoder_inputs') and len(dataset.decoder_inputs) > 0:
        dec_lengths = [len(tokens) for tokens in dataset.decoder_inputs]
        mean_dec_length = sum(dec_lengths) / len(dec_lengths)
    else:
        mean_dec_length = None
    
    # Vocabulary size (encoder - unique token IDs used)
    enc_vocab = set()
    for tokens in dataset.encoder_inputs:
        enc_vocab.update(tokens.tolist())
    
    # Vocabulary size (decoder - unique token IDs used)
    if hasattr(dataset, 'decoder_inputs') and len(dataset.decoder_inputs) > 0:
        dec_vocab = set()
        for tokens in dataset.decoder_inputs:
            dec_vocab.update(tokens.tolist())
        vocab_size_dec = len(dec_vocab)
    else:
        vocab_size_dec = None
    
    return {
        'num_examples': len(dataset),
        'mean_enc_length': mean_enc_length,
        'mean_dec_length': mean_dec_length,
        'vocab_size_enc': len(enc_vocab),
        'vocab_size_dec': vocab_size_dec
    }

print("="*70)
print("DATA STATISTICS - BEFORE PREPROCESSING (Raw Text)")
print("="*70)

# Compute statistics for train and dev BEFORE preprocessing
train_stats_before = compute_statistics_before('data', 'train')
dev_stats_before = compute_statistics_before('data', 'dev')

# Print BEFORE table
print(f"\n{'Statistics Name':<40} {'Train':<15} {'Dev':<15}")
print("-"*70)
print(f"{'Number of examples':<40} {train_stats_before['num_examples']:<15} {dev_stats_before['num_examples']:<15}")
print(f"{'Mean sentence length (words)':<40} {train_stats_before['mean_nl_length']:<15.2f} {dev_stats_before['mean_nl_length']:<15.2f}")
print(f"{'Mean SQL query length (words)':<40} {train_stats_before['mean_sql_length']:<15.2f} {dev_stats_before['mean_sql_length']:<15.2f}")
print(f"{'Vocabulary size (natural language)':<40} {train_stats_before['vocab_size_nl']:<15} {dev_stats_before['vocab_size_nl']:<15}")
print(f"{'Vocabulary size (SQL)':<40} {train_stats_before['vocab_size_sql']:<15} {dev_stats_before['vocab_size_sql']:<15}")
print("="*70)

print("\n" + "="*70)
print("LOADING TOKENIZED DATASETS...")
print("="*70)

# Load datasets (this does the preprocessing/tokenization)
try:
    train_dataset = T5Dataset('data', 'train')
    print(f"✓ Training set loaded")
except Exception as e:
    print(f"✗ Error loading training set: {e}")
    import traceback
    traceback.print_exc()
    train_dataset = None

try:
    dev_dataset = T5Dataset('data', 'dev')
    print(f"✓ Dev set loaded")
except Exception as e:
    print(f"✗ Error loading dev set: {e}")
    dev_dataset = None

try:
    test_dataset = T5Dataset('data', 'test')
    print(f"✓ Test set loaded")
except Exception as e:
    print(f"✗ Error loading test set: {e}")
    test_dataset = None

print("\n" + "="*70)
print("DATA STATISTICS - AFTER PREPROCESSING (Tokenized)")
print("="*70)

# Compute statistics AFTER preprocessing
if train_dataset and dev_dataset:
    train_stats_after = compute_statistics_after(train_dataset)
    dev_stats_after = compute_statistics_after(dev_dataset)
    
    # Print AFTER table
    print(f"\n{'Statistics Name':<40} {'Train':<15} {'Dev':<15}")
    print("-"*70)
    print(f"{'Number of examples':<40} {train_stats_after['num_examples']:<15} {dev_stats_after['num_examples']:<15}")
    print(f"{'Mean encoder length (tokens)':<40} {train_stats_after['mean_enc_length']:<15.2f} {dev_stats_after['mean_enc_length']:<15.2f}")
    print(f"{'Mean decoder length (tokens)':<40} {train_stats_after['mean_dec_length']:<15.2f} {dev_stats_after['mean_dec_length']:<15.2f}")
    print(f"{'Vocabulary size (encoder tokens)':<40} {train_stats_after['vocab_size_enc']:<15} {dev_stats_after['vocab_size_enc']:<15}")
    print(f"{'Vocabulary size (decoder tokens)':<40} {train_stats_after['vocab_size_dec']:<15} {dev_stats_after['vocab_size_dec']:<15}")
    print("="*70)

# Show example
print("\n" + "="*70)
print("EXAMPLE FROM TRAINING SET")
print("="*70)
if train_dataset and len(train_dataset) > 0:
    example = train_dataset[0]
    print(f"\nFirst example:")
    print(f"  Encoder input shape: {example[0].shape}")
    print(f"  Decoder input shape: {example[1].shape}")
    print(f"  Decoder target shape: {example[2].shape}")
    print(f"\n  Encoder tokens (first 10): {example[0][:10].tolist()}")
    print(f"  Decoder input tokens (first 10): {example[1][:10].tolist()}")
    print(f"  Decoder target tokens (first 10): {example[2][:10].tolist()}")