import pdfplumber
import pandas as pd
import requests
import json
import os
import re
from openai import OpenAI


'''
此代码用于识别《中国手语概论.pdf》，并生成中国手语概论QA对数据集。
调用飞浆AI Studio一键部署的Deepseek R1蒸馏模型去合成一份有效的数据集。）

注意！！！
使用前先填入自己一键部署Deepseek服务的key。
'''

# 替换原有的API调用部分
client = OpenAI(
    api_key="", # 自己填入自己的key
    base_url="https://api-r1i1sem25807m8z7.aistudio-app.com/v1"
)

# 提取PDF内容
def extract_pdf_text(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"提取PDF内容时出错: {e}")
        return None

# 使用Deepseek模型生成QA对
def generate_qa_pairs(text, num_pairs=15):
    # 限制文本长度，防止超出模型上下文窗口
    max_text_length = 8000  # 根据模型的具体限制调整
    if len(text) > max_text_length:
        print(f"文本太长({len(text)}字符)，截断至{max_text_length}字符")
        text = text[:max_text_length]
    
    prompt = f"""
    请根据以下关于中国手语语言学的文本内容，生成{num_pairs}个问答对。
    每个问答对应包含一个问题和相应的答案，问题应该涵盖文本中的重要概念、定义、历史发展和研究现状等方面。
    答案应从文本中提取，尽可能详细全面，不要添加任何其他文本或解释。
    
    文本内容:
    {text}
    
    请严格按照以下JSON格式返回结果，不要添加任何其他文本或解释:
    [
        {{"question": "问题1", "answer": "答案1"}},
        {{"question": "问题2", "answer": "答案2"}},
        ...
    ]
    """
    
    completion = client.chat.completions.create(
        model="deepseek-r1:1.5b",
        temperature=0.6,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    qa_text = completion.choices[0].message.content
    
    # 打印API响应的状态和结构，帮助调试
    print(f"API响应状态码: {completion.status_code}")
    print(f"API响应内容类型: {completion.headers.get('Content-Type')}")
    
    if "choices" not in completion or len(completion["choices"]) == 0:
        print(f"API响应缺少choices字段: {completion}")
        return []
        
    print(f"API返回的内容长度: {len(qa_text)} 字符")
    print("API返回的内容前100字符: " + qa_text[:100])
    
    # 尝试提取JSON部分
    json_pattern = r'\[\s*\{.*\}\s*\]'
    json_matches = re.search(json_pattern, qa_text, re.DOTALL)
    
    if json_matches:
        qa_text = json_matches.group(0)
        print("成功提取JSON部分")
    else:
        print("无法从响应中提取JSON部分，尝试直接解析")
    
    # 解析JSON响应
    qa_pairs = json.loads(qa_text)
    print(f"成功解析JSON，获取到 {len(qa_pairs)} 个QA对")
    return qa_pairs

# 将QA对保存为Excel文件
def save_qa_to_excel(qa_pairs, output_path):
    if not qa_pairs:
        print("没有QA对可保存")
        return
        
    df = pd.DataFrame(qa_pairs)
    df.to_excel(output_path, index=False)
    print(f"QA文档已保存至: {output_path}")

# 使用分批处理长文本
def process_long_text(text, chunk_size=3000, overlap=500, num_pairs_per_chunk=3):
    # 将长文本分成多个块
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text) and end - start > overlap:
            # 尝试在句子或段落边界处分割
            for i in range(min(end, len(text) - 1), start + chunk_size - overlap, -1):
                if text[i] in ['.', '。', '!', '！', '?', '？', '\n'] and text[i+1] in [' ', '\n']:
                    end = i + 1
                    break
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    
    print(f"文本已分割为 {len(chunks)} 个块")
    
    # 为每个块生成QA对
    all_qa_pairs = []
    for i, chunk in enumerate(chunks):
        print(f"处理第 {i+1}/{len(chunks)} 个文本块...")
        chunk_qa_pairs = generate_qa_pairs(chunk, num_pairs=25)
        all_qa_pairs.extend(chunk_qa_pairs)
    
    return all_qa_pairs

# 主函数
def main():
    pdf_path = "中国手语概述——从历史发展到现代研究的全面分析.pdf"
    output_path = "中国手语概述QA文档.xlsx"
    
    # 提取PDF内容
    print("正在提取PDF内容...")
    pdf_text = extract_pdf_text(pdf_path)
    
    if pdf_text:
        print(f"PDF内容提取成功，共 {len(pdf_text)} 字符")
        
        # 判断文本是否太长，如果太长则使用分批处理
        if len(pdf_text) > 3000:
            print("文本较长，使用分批处理...")
            qa_pairs = process_long_text(pdf_text, num_pairs_per_chunk=25)
        else:
            print("正在生成QA对...")
            qa_pairs = generate_qa_pairs(pdf_text, num_pairs=25)
        
        if qa_pairs:
            save_qa_to_excel(qa_pairs, output_path)
        else:
            print("生成QA对失败")
            
            # 尝试使用备用方法
            print("尝试使用备用方法生成简单QA对...")
            simple_qa_pairs = [
                {"question": "什么是中国手语？", "answer": "中国手语是中国聋人社区使用的视觉-手势语言系统。"},
                {"question": "中国手语的研究历史如何？", "answer": "中国手语的系统研究始于20世纪，经历了从初步记录到语言学分析的发展过程。"}
            ]
            save_qa_to_excel(simple_qa_pairs, output_path)
    else:
        print("PDF内容提取失败")

if __name__ == "__main__":
    main() 