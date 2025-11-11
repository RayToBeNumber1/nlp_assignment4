import os
from transformers import T5TokenizerFast
from load_data import load_lines

def compute_statistics():
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    data_folder = 'data'
    
    stats = {}
    
    for split in ['train', 'dev']:
        # 加载数据
        nl_path = os.path.join(data_folder, f'{split}.nl')
        sql_path = os.path.join(data_folder, f'{split}.sql')
        
        nl_sentences = load_lines(nl_path)
        sql_queries = load_lines(sql_path)
        
        # Tokenize
        nl_tokenized = tokenizer(nl_sentences, add_special_tokens=True)
        sql_tokenized = tokenizer(sql_queries, add_special_tokens=True)
        
        # 计算统计
        num_examples = len(nl_sentences)
        
        # 原始文本长度（单词数）
        nl_word_lengths = [len(sent.split()) for sent in nl_sentences]
        sql_word_lengths = [len(query.split()) for query in sql_queries]
        
        # Tokenized长度
        nl_token_lengths = [len(ids) for ids in nl_tokenized['input_ids']]
        sql_token_lengths = [len(ids) for ids in sql_tokenized['input_ids']]
        
        # 词汇表大小
        nl_vocab = set()
        for sent in nl_sentences:
            nl_vocab.update(sent.lower().split())
        
        sql_vocab = set()
        for query in sql_queries:
            sql_vocab.update(query.split())
        
        stats[split] = {
            'num_examples': num_examples,
            'mean_nl_words': sum(nl_word_lengths) / len(nl_word_lengths),
            'mean_sql_words': sum(sql_word_lengths) / len(sql_word_lengths),
            'mean_nl_tokens': sum(nl_token_lengths) / len(nl_token_lengths),
            'mean_sql_tokens': sum(sql_token_lengths) / len(sql_token_lengths),
            'nl_vocab_size': len(nl_vocab),
            'sql_vocab_size': len(sql_vocab),
        }
    
    # 打印Table 1 (预处理前 - 基于单词)
    print("=" * 60)
    print("Table 1: Data statistics before preprocessing (word-based)")
    print("=" * 60)
    print(f"{'Statistics Name':<35} {'Train':<12} {'Dev':<12}")
    print("-" * 60)
    print(f"{'Number of examples':<35} {stats['train']['num_examples']:<12} {stats['dev']['num_examples']:<12}")
    print(f"{'Mean sentence length (words)':<35} {stats['train']['mean_nl_words']:<12.2f} {stats['dev']['mean_nl_words']:<12.2f}")
    print(f"{'Mean SQL query length (words)':<35} {stats['train']['mean_sql_words']:<12.2f} {stats['dev']['mean_sql_words']:<12.2f}")
    print(f"{'Vocabulary size (natural language)':<35} {stats['train']['nl_vocab_size']:<12} {stats['dev']['nl_vocab_size']:<12}")
    print(f"{'Vocabulary size (SQL)':<35} {stats['train']['sql_vocab_size']:<12} {stats['dev']['sql_vocab_size']:<12}")
    print()
    
    # 打印Table 2 (预处理后 - 基于token)
    print("=" * 60)
    print("Table 2: Data statistics after preprocessing (token-based)")
    print("=" * 60)
    print(f"{'Statistics Name':<35} {'Train':<12} {'Dev':<12}")
    print("-" * 60)
    print(f"{'Mean sentence length (tokens)':<35} {stats['train']['mean_nl_tokens']:<12.2f} {stats['dev']['mean_nl_tokens']:<12.2f}")
    print(f"{'Mean SQL query length (tokens)':<35} {stats['train']['mean_sql_tokens']:<12.2f} {stats['dev']['mean_sql_tokens']:<12.2f}")
    print(f"{'Vocabulary size (T5 tokenizer)':<35} {tokenizer.vocab_size:<12} {tokenizer.vocab_size:<12}")
    print()

if __name__ == '__main__':
    compute_statistics()
