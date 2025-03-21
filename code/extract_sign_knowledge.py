import pdfplumber
import pandas as pd
import requests
import json
import os
import re
from PIL import Image
import fitz  # PyMuPDF
import io
import numpy as np
import time
import base64
from openai import OpenAI


'''
此代码用于识别手语语言学教材，并生成手语语言学知识点数据集。
因无法直接获取教材的文字版，该教材为pdf格式，无法直接提取文字，
需先使用百度OCR：通用文字识别高精度版识别所有文字，并调用飞浆AI Studio一键部署的Deepseek R1蒸馏模型去合成一份有效的数据集。）

注意！！！
使用前先填入百度ocr的两个key以及自己一键部署Deepseek服务的key。

'''

# 使用百度OCR API识别PDF页面内容
def extract_text_with_baidu_ocr(pdf_path, start_page=15, end_page=None, save_path="ocr_result1.txt"):
    # 检查是否已有保存的OCR结果
    if os.path.exists(save_path):
        print(f"发现已保存的OCR结果: {save_path}")
        user_input = input("是否使用已保存的OCR结果? (y/n): ").strip().lower()
        if user_input == 'y':
            with open(save_path, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
                print(f"已加载OCR结果，共 {len(ocr_text)} 字符")
                return ocr_text
    
    print(f"使用百度OCR API提取PDF内容，从第{start_page+1}页开始...")
    
    # 百度OCR API配置  请配置“通用文字识别高精度版”（标准版提取文字的效果很差）
    # 先去开通服务，创建应用，再复制key：https://console.bce.baidu.com/ai-engine/old/#/ai/ocr/app/list
    API_KEY = "填入自己的key"
    SECRET_KEY = "填入对应的key"
    
    # 获取access_token
    token_url = "https://aip.baidubce.com/oauth/2.0/token"
    token_params = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": SECRET_KEY  # 确保这里有SECRET_KEY
    }
    
    try:
        token_response = requests.post(token_url, params=token_params)
        token_response.raise_for_status()
        access_token = token_response.json().get("access_token")
        print("access_token:", access_token)
        if not access_token:
            print("获取百度OCR access_token失败")
            return None
    except Exception as e:
        print(f"获取百度OCR access_token时出错: {e}")
        return None
    
    # OCR API URL
    ocr_url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token={access_token}"
    
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    total_pages = pdf_document.page_count
    
    if end_page is None:
        end_page = total_pages - 1
    
    # 确保页码在有效范围内
    start_page = max(0, min(start_page, total_pages - 1))
    end_page = max(start_page, min(end_page, total_pages - 1))
    
    print(f"PDF总页数: {total_pages}, 处理范围: {start_page+1}-{end_page+1}页")
    
    all_text = ""
    
    # 创建临时文件，用于保存中间结果
    temp_save_path = save_path + ".temp"
    
    # 遍历指定页面范围
    for page_num in range(start_page, end_page + 1):
        print(f"正在处理第{page_num+1}页...")
        
        # 获取页面
        page = pdf_document[page_num]
        
        # 将页面渲染为图像
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 放大2倍以提高OCR质量
        img_bytes = pix.tobytes("png")
        
        # 检查图像大小
        img_size_mb = len(img_bytes) / (1024 * 1024)
        if img_size_mb > 3.5:  # 百度OCR限制为4MB，留一些余量
            print(f"图像太大 ({img_size_mb:.2f}MB)，降低分辨率...")
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 降低分辨率
            img_bytes = pix.tobytes("png")
            img_size_mb = len(img_bytes) / (1024 * 1024)
            print(f"调整后的图像大小: {img_size_mb:.2f}MB")
        
        # 将图像编码为base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # 发送OCR请求
        ocr_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        ocr_data = {
            'image': img_base64,
            'language_type': 'CHN_ENG',  # 中英文混合
            'detect_direction': 'true',  # 检测文字方向
            'paragraph': 'true',  # 输出段落信息
            'probability': 'true'  # 输出置信度
        }
        
        try:
            ocr_response = requests.post(ocr_url, headers=ocr_headers, data=ocr_data)
            ocr_response.raise_for_status()
            ocr_result = ocr_response.json()
            
            # 提取文本
            page_text = ""
            if 'words_result' in ocr_result:
                for word_info in ocr_result['words_result']:
                    page_text += word_info['words'] + " "
            
            # 添加页码信息
            page_content = f"\n\n--- 第{page_num+1}页 ---\n\n{page_text}"
            print("已提取：\n", page_content)
            all_text += page_content
            
            # 保存中间结果到临时文件
            with open(temp_save_path, 'a', encoding='utf-8') as f:
                f.write(page_content)
            
            # 百度OCR API有QPS限制，添加延迟
            time.sleep(1)
            
        except Exception as e:
            print(f"处理第{page_num+1}页时出错: {e}")
            # 如果出错，继续处理下一页
            continue
    
    pdf_document.close()
    
    # 保存完整OCR结果
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(all_text)
    print(f"OCR结果已保存至: {save_path}")
    
    # 删除临时文件
    if os.path.exists(temp_save_path):
        os.remove(temp_save_path)
    
    return all_text

# 从文件加载OCR结果
def load_ocr_result(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ocr_text = f.read()
            print(f"已从{file_path}加载OCR结果，共 {len(ocr_text)} 字符")
            return ocr_text
    except Exception as e:
        print(f"加载OCR结果时出错: {e}")
        return None

# 使用Deepseek API提取和整理手语知识
def extract_sign_language_knowledge(text, output_path="sign_language_knowledge.json"):
    # 检查是否已有保存的提取结果
    if os.path.exists(output_path):
        print(f"发现已保存的提取结果: {output_path}")
        user_input = input("是否使用已保存的提取结果? (y/n): ").strip().lower()
        if user_input == 'y':
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
                    print(f"已加载提取结果，共 {len(entries)} 个知识条目")
                    return entries
            except Exception as e:
                print(f"加载提取结果时出错: {e}")
                # 继续执行提取流程
    
    # 修改API配置
    client = OpenAI(
        api_key="填入自己的key",
        base_url="https://api-r1i1sem25807m8z7.aistudio-app.com/v1"
    )
    
    # 由于文本可能很长，我们需要分批处理
    max_chunk_size = 1500  # 块大小
    overlap = 250  # 重叠区域大小
    
    # 创建带有重叠的文本块
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        
        # 如果不是最后一块，尝试在句子或段落边界处分割
        if end < len(text):
            # 在重叠区域内寻找合适的分割点（句号、换行等）
            for i in range(end, max(end - overlap, start), -1):
                if text[i] in ['.', '。', '!', '！', '?', '？', '\n'] and i+1 < len(text) and text[i+1] in [' ', '\n']:
                    end = i + 1
                    break
        
        chunks.append(text[start:end])
        # 下一块的起始位置减去重叠区域大小
        start = end - overlap if end < len(text) else end
    
    print(f"文本已分割为 {len(chunks)} 个块，每块大小约 {max_chunk_size} 字符，重叠区域 {overlap} 字符")
    
    all_entries = []
    temp_output_path = output_path + ".temp"
    
    for i, chunk in enumerate(chunks):
        print(f"处理文本块 {i+1}/{len(chunks)}...")
        print("正在处理：\n", chunk)
        
        # 为每个块创建提示
        prompt = f"""
请从以下中国手语教材内容中提取手语语言学知识，按照以下格式组织：
- 小标题：提取文本中的小标题或主题
- 具体内容：提取与小标题相关的详细内容和解释
- 其他相关信息：提取与主题相关的补充信息、例子或应用场景

例如，当看到类似以下内容：
(二)时间线
手语中很少用动词时态或用以表达时态的动词屈折变化，它是利用空间维度连同用来表达特定时间概念的词汇一起创造出它所特有的语法表达机制来表达时间框架。
时间在此基础上可被划分为三个相对的时间类型...

可提取为：
{{
  "小标题": "时间线",
  "具体内容": "手语中很少用动词时态或用以表达时态的动词屈折变化，它是利用空间维度连同用来表达特定时间概念的词汇一起创造出它所特有的语法表达机制来表达时间框架。时间在此基础上可被划分为三个相对的时间类型:交际本身发生的时间--现在时间框架;交际发生之前的时间--过去时间框架;交际发生之后的时间--将来时间框架。...",
  "其他相关信息": "时间线是手语语法的重要组成部分，与口语的时态表达方式有显著不同。"
}}

请仔细分析文本，提取所有相关知识点（一个文本内容可以提取多个知识点）。如果某个部分信息不存在，则填写空字符串。
忽略非知识性内容，如页码、参考文献等。

文本内容:
{chunk}

请严格遵循以下JSON数组格式返回结果:
[
  {{"小标题": "标题1", "具体内容": "内容1", "其他相关信息": "信息1"}},
  {{"小标题": "标题2", "具体内容": "内容2", "其他相关信息": "信息2"}}
]
"""
        
        completion = client.chat.completions.create(
            model="deepseek-r1:1.5b",
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        content = completion.choices[0].message.content
        
        # 提取JSON部分
        json_pattern = r'\[\s*\{.*\}\s*\]'
        json_matches = re.search(json_pattern, content, re.DOTALL)
        
        if not json_matches:
            print("无法从响应中提取JSON")
            print("API返回内容:", content[:200])  # 打印部分内容以便调试
            continue
        
        json_str = json_matches.group(0)
        
        # 解析JSON
        try:
            entries = json.loads(json_str)
            print(f"从当前块中提取了 {len(entries)} 个知识条目")
            all_entries.extend(entries)
            
            # 保存中间结果
            with open(temp_output_path, 'w', encoding='utf-8') as f:
                json.dump(all_entries, f, ensure_ascii=False, indent=2)
            
        except json.JSONDecodeError as je:
            print(f"JSON解析错误: {je}")
            print(f"问题JSON: {json_str[:100]}...")
            # 尝试修复常见的JSON格式问题
            fixed_json = json_str.replace("'", '"').replace("，", ",")
            try:
                entries = json.loads(fixed_json)
                print(f"修复后成功解析，提取了 {len(entries)} 个知识条目")
                all_entries.extend(entries)
            except:
                print("修复JSON后仍然解析失败")
        
        # 添加延迟，避免API请求过于频繁
        time.sleep(2)
        
    # 去重
    unique_entries = []
    seen_entries = set()
    
    for entry in all_entries:
        # 创建唯一标识
        key = entry.get("小标题", "")
        if key not in seen_entries and key.strip() != "":
            seen_entries.add(key)
            unique_entries.append(entry)
    
    # 保存最终结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_entries, f, ensure_ascii=False, indent=2)
    print(f"提取结果已保存至: {output_path}")
    
    # 删除临时文件
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    
    return unique_entries

# 处理更小的文本块
def process_smaller_knowledge_chunk(chunk, index, total, api_url, api_key, all_entries):
    print(f"处理更小的文本块 {index+1}/{total}...")
    
    prompt = f"""
提取以下文本中的手语语言学知识，按照以下格式组织（一个文本内容可以提取多个知识点）：
- 小标题：提取文本中的小标题或主题
- 具体内容：提取与小标题相关的详细内容和解释
- 其他相关信息：提取与主题相关的补充信息、例子或应用场景

文本内容:
{chunk}

请以JSON格式返回结果:
[{{"小标题": "标题", "具体内容": "内容", "其他相关信息": "信息"}}]
"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 修改这里 - 使用正确的模型ID和参数
    data = {
        "model": "Qwen/Qwen2.5-72B-Instruct",  # 使用有效的模型ID
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 8192  # 减小token数量
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            
            # 提取JSON部分
            json_pattern = r'\[\s*\{.*\}\s*\]'
            json_matches = re.search(json_pattern, content, re.DOTALL)
            
            if json_matches:
                json_str = json_matches.group(0)
                try:
                    entries = json.loads(json_str)
                    print(f"从更小的块中提取了 {len(entries)} 个知识条目")
                    all_entries.extend(entries)
                except json.JSONDecodeError:
                    print("JSON解析错误，尝试修复...")
                    # 尝试修复JSON
                    fixed_json = json_str.replace("'", '"').replace("，", ",")
                    try:
                        entries = json.loads(fixed_json)
                        print(f"修复后成功解析，提取了 {len(entries)} 个知识条目")
                        all_entries.extend(entries)
                    except:
                        print("修复JSON后仍然解析失败")
        
        # 添加延迟，避免API请求过于频繁
        time.sleep(2)
        
    except Exception as e:
        print(f"处理更小的文本块时出错: {e}")

# 将提取的知识条目保存为Excel文件
def save_to_excel(entries, output_path):
    if not entries:
        print("没有知识条目可保存")
        return False
        
    df = pd.DataFrame(entries)
    
    # 确保所有必要的列都存在
    required_columns = ["小标题", "具体内容", "其他相关信息"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
    
    # 过滤掉空标题
    df = df[df["小标题"].str.strip() != ""]
    
    # 按标题排序
    df = df.sort_values("小标题")
    
    # 重新排列列顺序
    df = df[required_columns]
    
    df.to_excel(output_path, index=False)
    print(f"已将 {len(df)} 个手语知识条目保存至: {output_path}")
    return True

# 主函数
def main():
    pdf_path = "中国手语语言学.pdf"
    ocr_result_path = "ocr_result3.txt"
    knowledge_path = "sign_language_knowledge.json"
    output_excel_path = "中国手语知识数据集.xlsx"
    
    # 步骤1: 提取PDF内容或加载已有OCR结果
    if os.path.exists(ocr_result_path):
        print(f"发现已保存的OCR结果: {ocr_result_path}")
        user_input = input("是否使用已保存的OCR结果? (y/n): ").strip().lower()
        if user_input == 'y':
            ocr_text = load_ocr_result(ocr_result_path)
        else:
            print("开始提取PDF内容...")
            ocr_text = extract_text_with_baidu_ocr(pdf_path, start_page=14, save_path=ocr_result_path)
    else:
        print("开始提取PDF内容...")
        ocr_text = extract_text_with_baidu_ocr(pdf_path, start_page=14, save_path=ocr_result_path)
    
    if not ocr_text:
        print("PDF内容提取失败")
        return
    
    # 步骤2: 提取手语知识或加载已有提取结果
    print("开始提取手语知识...")
    knowledge_entries = extract_sign_language_knowledge(ocr_text, output_path=knowledge_path)
    
    # 步骤3: 保存为Excel
    if knowledge_entries:
        save_to_excel(knowledge_entries, output_excel_path)
    else:
        print("未能提取到手语知识")

if __name__ == "__main__":
    main() 