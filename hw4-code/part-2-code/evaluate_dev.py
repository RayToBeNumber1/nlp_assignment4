import os
import argparse
import torch
from tqdm import tqdm
from transformers import T5TokenizerFast

from t5_utils import initialize_model
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def evaluate_dev_set(checkpoint_path, experiment_name='ft_experiment'):
    """
    使用训练好的模型评估dev set
    """
    # 创建临时args来加载模型
    class Args:
        finetune = True
        checkpoint_dir = os.path.dirname(checkpoint_path)
    
    args = Args()
    
    # 加载模型
    print(f"Loading model from {checkpoint_path}")
    model = initialize_model(args)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 加载dev数据
    print("Loading dev data...")
    _, dev_loader, _ = load_t5_data(16, 16)
    
    # 生成SQL查询
    print("Generating SQL queries...")
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    all_generated_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_input in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # 生成SQL
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=768,
                num_beams=4,
                early_stopping=True
            )
            
            # 解码
            generated_queries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_generated_queries.extend(generated_queries)
    
    # 保存结果
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = f'results/t5_ft_{experiment_name}_dev_eval.sql'
    model_record_path = f'records/t5_ft_{experiment_name}_dev_eval.pkl'
    
    print("Saving queries and computing database records...")
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)
    
    # 计算metrics
    print("Computing metrics...")
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    
    error_rate = len([msg for msg in error_msgs if msg]) / len(error_msgs) if error_msgs else 0
    
    # 打印结果
    print("\n" + "="*60)
    print("Dev Set Evaluation Results")
    print("="*60)
    print(f"Record F1:       {record_f1*100:.2f}%")
    print(f"Record EM:       {record_em*100:.2f}%")
    print(f"SQL EM:          {sql_em*100:.2f}%")
    print(f"SQL Error Rate:  {error_rate*100:.2f}%")
    print("="*60)
    
    return record_f1, record_em, sql_em, error_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint (e.g., checkpoints/ft_experiments/improved/best_model.pt)')
    parser.add_argument('--experiment_name', type=str, default='ft_experiment',
                        help='Name for output files')
    
    args = parser.parse_args()
    
    evaluate_dev_set(args.checkpoint, args.experiment_name)
