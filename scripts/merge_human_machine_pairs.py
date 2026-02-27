import json
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_directories(extracted_dir: str, polished_dir: str, output_file: str):
    """
    遍历提取的人类文本目录与生成的 AI 文本目录，按句子一一对应合并为 .jsonl 格式的训练数据集。
    """
    extracted_path = Path(extracted_dir)
    polished_path = Path(polished_dir)
    
    if not extracted_path.exists():
        logging.error(f"提取文本目录不存在: {extracted_path}")
        return
        
    if not polished_path.exists():
        logging.error(f"AI 润色文本目录不存在: {polished_path}")
        return
        
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    extracted_files = list(extracted_path.glob('**/*.json'))
    logging.info(f"在 extracted 目录中找到 {len(extracted_files)} 个 JSON 文件。")
    
    total_pairs = 0
    skipped_files = 0
    
    with open(out_path, 'w', encoding='utf-8') as out_f:
        for ex_file in extracted_files:
            rel_path = ex_file.relative_to(extracted_path)
            pol_file = polished_path / rel_path
            
            if not pol_file.exists():
                logging.warning(f"对应的 AI 文本文件不存在: {rel_path}")
                skipped_files += 1
                continue
                
            try:
                with open(ex_file, 'r', encoding='utf-8') as f:
                    human_texts = json.load(f)
                with open(pol_file, 'r', encoding='utf-8') as f:
                    machine_texts = json.load(f)
                    
                if not isinstance(human_texts, list) or not isinstance(machine_texts, list):
                    logging.warning(f"文件格式错误（非 JSON 数组）: {rel_path}")
                    skipped_files += 1
                    continue
                    
                if len(human_texts) != len(machine_texts):
                    logging.warning(f"句子数量不匹配: {rel_path} (Human: {len(human_texts)}, Machine: {len(machine_texts)})")
                    
                min_len = min(len(human_texts), len(machine_texts))
                
                for i in range(min_len):
                    human_sent = human_texts[i].strip()
                    machine_sent = machine_texts[i].strip()
                    
                    if human_sent and machine_sent:
                        record = {
                            "human": human_sent,
                            "machine": machine_sent
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        total_pairs += 1
                        
            except Exception as e:
                logging.error(f"处理文件对 {rel_path} 时发生错误: {e}")
                skipped_files += 1
                
    logging.info("=" * 50)
    logging.info(f"合并完成！共生成 {total_pairs} 对训练语料。")
    logging.info(f"跳过的匹配失败文件数: {skipped_files}")
    logging.info(f"输出的 Dataset 文件路径: {out_path.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并人类提取文本和 AI 润色文本，生成 DetectAnyLLM 训练可用的 .jsonl 数据集。")
    parser.add_argument("--extracted_dir", type=str, required=True, help="第一步：人类提取文本所在的目录")
    parser.add_argument("--polished_dir", type=str, required=True, help="第二步：AI 润色文本所在的目录")
    parser.add_argument("--output_file", type=str, default="data/train_pairs.jsonl", help="输出的 JSONL 文件路径")
    
    args = parser.parse_args()
    
    merge_directories(args.extracted_dir, args.polished_dir, args.output_file)
