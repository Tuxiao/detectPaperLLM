import json
import random
import re
from pathlib import Path
from typing import List, Dict

def full_to_half(text: str) -> str:
    """
    全角转半角，常用于清理各种由于排版引起的奇怪字符
    例如将“ｈｔｔｐ：”转换为“http:”
    """
    result = []
    for char in text:
        code = ord(char)
        # 全角空格直接转换
        if code == 12288:
            code = 32
        # 全角字符（除空格外）在其对应的半角字符之间相差 65248
        elif 65281 <= code <= 65374:
            code -= 65248
        result.append(chr(code))
    return "".join(result)

def clean_text(text: str) -> str:
    """
    清洗文本：
    1. 全角字符转半角字符（如 ｈｔｔｐ -> http）。
    2. 去除特定的 unicode 转换残留，例如 /uniFF08, /uniFF09 等。
    3. 去除所有的不可见字符（空格、换行符、回车符、制表符等）。
    4. 去除前后的空白。
    按照 DetectAnyLLM 论文中 4.1 Benchmark Construction 的要求（remove all the '\\n' character）。
    """
    if not text:
        return ""
        
    # 0. 全角字符转换为半角（重点解决特殊英文字母变宽和全角标点的问题）
    text = full_to_half(text)
    
    # 将一些特殊的类似字符替换为标准字符，如 ∥ (U+2225) 换为 //
    text = text.replace("∥", "//")
    
    # 1. 移除特定的乱码序列，如 /uniff08, /uniff09, /uniff3b 等
    # 这些通常是特定 PDF 解析工具输出的 Unicode 标记残留
    # 由于转换为了半角并有可能变小写，用 re.IGNORECASE 匹配
    cleaned = re.sub(r'/uni[0-9A-Fa-f]{4}', '', text, flags=re.IGNORECASE)
    
    # 2. 替换所有的空白字符（包括全角/半角空格、\n, \r, \t等）为空字符串
    # 这个包含了去除诸如 " （ 一 ）" 这种里面的多余空格
    cleaned = re.sub(r'\s+', '', cleaned)
    
    return cleaned

def is_reference_segment(text: str) -> bool:
    """
    判断片段是否高度疑似参考文献或网址引用。
    如果疑似，返回 True，整段将不被采用。
    """
    # 参考文献特征 1：大量包含典型的英文期刊引用格式、卷期号特征
    # 比如 "Vol.", "Journal of", "et al.", "pp.", "1008-1020" 等
    ref_keywords = ['etal.', 'journalof', 'vol.', 'pp.', '[j]', '[m]', '[eb/ol]', '[n]', '[d]']
    
    # 将文本转小写进行匹配，提高鲁棒性
    lower_text = text.lower()
    
    # 如果包含常见的网址格式
    if 'http://' in lower_text or 'https://' in lower_text or 'www.' in lower_text:
        return True
        
    # 如果包含多个典型的参考文献标识符
    keyword_matches = sum(1 for kw in ref_keywords if kw in lower_text)
    if keyword_matches >= 2: # 包含2个或以上典型参考文献特征词
        return True
        
    # 参考文献特征 2：英文文献通常是一堆名字逗号分隔，并带有年份
    # 比如 "Surya Nepal, et al.", "2014," 
    # 可以用比较强的正则匹配，这里简单用字母占比和年份判断。纯英文段落如果在中文论文里出现，大概率是参考文献或摘要。
    # 也可以检查是否由 "[数字]" 开头，如 "[1] xxx"
    if re.search(r'^\[\d+\]', text) or (re.search(r'^\d+\.', text) and not re.search(r'[\u4e00-\u9fa5]', text[:20])):
         return True
         
    # 特征 3：大量类似 "10.1002/", "1008-1020" 这样的 DOI 或页码/期号模式，且中文很少
    cnt_chinese = len(re.findall(r'[\u4e00-\u9fa5]', text))
    if len(text) > 0 and (cnt_chinese / len(text)) < 0.2:
        # 如果一段 100-200 字的内容里，中文字符占比不足 20%，这在纯中文论文中极可能是英文参考文献或英文摘要。可以安全滤除。
        return True
        
    # 特征 4：中文典型参考文献格式，如 "姓名.标题[J].期刊名,2010(3):1-5."
    # 匹配类似 ",2018,46(6):158-170." 或 ",2010(3):1-5." 或 ".2010(3)" 的正则特征
    # [J], [M] 虽然在 ref_keywords 中，但由于全角转半角和无空格化，有时候会漏掉或者没达到2个特征词
    if re.search(r',\d{4},\d+\(\d+\):\d+-\d+', text) or \
       re.search(r',\d{4}\(\d+\):\d+-\d+', text) or \
       re.search(r'\.\d{4}\(\d+\):\d+-\d+', text):
        return True

    return False

def split_into_segments(text: str, min_len: int = 100, max_len: int = 200) -> List[str]:
    """
    将长文本切分为长度在 min_len 到 max_len 之间的片段。
    采用按标点符号切分，累加句子直到满足长度要求的方式。
    """
    if not text:
        return []

    # 按照中文常见句末标点符号进行切分（保留标点符号）
    # 使用 () 捕获标点符号，这样 split 后标点也会作为元素保留在列表中
    sentences_with_punct = re.split(r'([。！？；.!?;])', text)
    
    # 重新组合句子和标点
    combined_sentences = []
    current_sentence = ""
    for item in sentences_with_punct:
        if re.match(r'[。！？；.!?;]', item):
            current_sentence += item
            combined_sentences.append(current_sentence)
            current_sentence = ""
        else:
            current_sentence += item
    if current_sentence: # 处理末尾没有标点的残句
        combined_sentences.append(current_sentence)
        
    segments = []
    current_segment = ""
    
    for sentence in combined_sentences:
        # 如果当前片段加上新句子超过最大长度
        if len(current_segment) + len(sentence) > max_len:
            # 如果当前片段已经大于最小长度，则直接把当前片段保存下来，新句子另起一段
            if len(current_segment) >= min_len:
                segments.append(current_segment)
                current_segment = sentence
            else:
                # 走到这里意味着：当前片段还不够最小长度，但加上新句子后超过了最大长度。
                # 这说明这个新句子非常长，或者是前面积累的片段加上一个长句子。
                # 【重要优化】为了保持句意完整，我们不再按 max_len 硬切分。
                # 宁可让这个片段稍微超出 max_len，也要保证句子完整。
                current_segment += sentence
                segments.append(current_segment)
                current_segment = ""
        else:
            current_segment += sentence
            
    # 处理文章结束时剩下的最后一个片段
    if current_segment and len(current_segment) >= min_len:
        segments.append(current_segment)
        
    # 二次过滤，放宽一点上限（比如允许到 250字 以容纳那些被完整保留的长句），
    # 但严格保证下限 min_len，避免丢掉符合语境的完整片段。
    valid_segments = [seg for seg in segments if min_len <= len(seg) <= max_len + 50]
    
    # 过滤掉参考文献和不良片段
    final_segments = []
    for seg in valid_segments:
        if not is_reference_segment(seg):
            final_segments.append(seg)
            
    return final_segments

def extract_text_from_paper(json_data: Dict) -> str:
    """
    从 JSON 格式的论文中提取所有核心文本内容。
    遍历 sections 中的 paragraphs、bullets、subsections，拼接成一整段长文本。
    """
    texts = []
    
    # 获取最外层的摘要
    abstract = json_data.get('abstract') or json_data.get('paperAbstract')
    if abstract:
        texts.append(abstract)

    def process_section(sec: dict):
        # 提取标题
        heading = sec.get('heading', '')
        if heading:
            # 如果标题没有标点结尾，强行加一个句号以便后续分句
            if not re.search(r'[。！？；.!?;]$', heading.strip()):
                heading += "。"
            texts.append(heading)
            
        # 提取段落
        paragraphs = sec.get('paragraphs', [])
        for para in paragraphs:
            texts.append(para)
            
        # 提取列举项
        bullets = sec.get('bullets', [])
        for bullet in bullets:
            texts.append(bullet)
            
        # 递归处理子章节
        subsections = sec.get('subsections', [])
        for sub in subsections:
            process_section(sub)

    sections = json_data.get('sections', [])
    for section in sections:
        process_section(section)
            
    # 将提取出的所有块拼接为一个完整的字符串
    full_text = "".join(texts)
    return full_text

def process_papers_to_dataset(input_dir: str, output_file: str, min_words: int = 100, max_words: int = 200, seed: int = 42):
    """
    处理目录下所有的论文 JSON 文件，提取内容并组装 JSONL。
    """
    random.seed(seed)
    
    input_path = Path(input_dir)
    json_files = list(input_path.glob('**/*.json'))
    print(f"找到 {len(json_files)} 个 JSON 文件待处理...")
    
    dataset_records = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 1. 递归提取出论文全文本
            raw_text = extract_text_from_paper(data)
            
            # 2. 清洗文本（完全去除特定乱码序列和所有空白字符）
            cleaned_text = clean_text(raw_text)
            
            # 3. 按语意和标点符号将长文本切分为 100 到 200 字之间的片段列表，并滤除参考文献
            valid_segments = split_into_segments(cleaned_text, min_len=min_words, max_len=max_words)
            
            # 4. 如果该论文能拉取出符合要求的片段，则从中随机挑取 "1" 段作为最终样本
            if valid_segments:
                selected_segment = random.choice(valid_segments)
                dataset_records.append({
                    "id": file_path.stem, # 论文名字作为 ID
                    "human": selected_segment
                })
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")
            
    # 5. 写入 JSONL 格式（每行一个 JSON 对象）
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for record in dataset_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"==================================================")
    print(f"处理完成！成功从 {len(json_files)} 篇论文中提取了 {len(dataset_records)} 条人类文本语料片段。")
    print(f"输出文件保存在：{output_file}")
    if dataset_records:
        print("\n【示例数据一条】：")
        print(json.dumps(dataset_records[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="将论文 JSON 文件处理清洗并切分为 DetectAnyLLM 训练集所需的人类原始片段。")
    parser.add_argument("--input_dir", type=str, default="data/grouped_pdfs_by_keywords_random200_around1m_paper_json", help="输入包含论文嵌套 JSON 结构的根目录")
    parser.add_argument("--output_file", type=str, default="data/human_texts.jsonl", help="输出准备送去机器扩写的源文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机抽取片段的种子")
    
    args = parser.parse_args()
    
    process_papers_to_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        seed=args.seed
    )
