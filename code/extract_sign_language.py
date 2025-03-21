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
此代码用于识别《中国手语.pdf》，并生成中国手语词目操作数据集。
因无法直接获取教材的文字版，该教材为pdf格式，无法直接提取文字，
需先使用百度OCR：通用文字识别高精度版识别所有文字，并调用飞浆AI Studio一键部署的Deepseek R1蒸馏模型去合成一份有效的数据集。）

注意！！！
使用前先填入百度ocr的两个key以及自己一键部署Deepseek服务的key。

'''


# 优化建议：并行处理，使用多线程或异步处理，提高处理速度。

# 使用百度OCR API识别PDF页面内容
def extract_text_with_baidu_ocr(pdf_path, start_page=16, end_page=None, save_path="ocr_result.txt"):
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
    
    # 百度OCR API配置  通用文字识别高精度版。需自行填入
    API_KEY = ""
    SECRET_KEY = ""
    
    # 获取access_token
    token_url = "https://aip.baidubce.com/oauth/2.0/token"
    token_params = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": SECRET_KEY
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
            print(page_content)
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

# 使用Deepseek API提取和整理手语词目
def extract_sign_language_entries(text, output_path="extracted_entries.json"):
    # 检查是否已有保存的提取结果
    if os.path.exists(output_path):
        print(f"发现已保存的提取结果: {output_path}")
        user_input = input("是否使用已保存的提取结果? (y/n): ").strip().lower()
        if user_input == 'y':
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
                    print(f"已加载提取结果，共 {len(entries)} 个词条")
                    return entries
            except Exception as e:
                print(f"加载提取结果时出错: {e}")
                # 继续执行提取流程
    
    client = OpenAI(
        api_key="",   # 自己填入
        base_url="https://api-r1i1sem25807m8z7.aistudio-app.com/v1"
    )
    
    # 由于文本可能很长，我们需要分批处理
    # 减小块大小以避免413错误，并添加重叠区域
    max_chunk_size = 800  # 块大小
    overlap = 50  # 重叠区域大小
    
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
        print("正在处理：", chunk)
        
        # 为每个块创建一个更简洁的提示
        prompt = f"""
请从以下中国手语教材内容中提取所有手语词目及其对应的手势操作描述。

例如，当看到类似"脸 lian face 一手五指并拢轻贴一下面颊部"、"G:食指伸直，指尖向左，其余四指握拳，手背朝外。"这样的内容时，应提取为：
"词目": "脸 lian face"
"操作": "一手五指并拢轻贴一下面颊部"
"词目": "G"
"操作": "食指伸直，指尖向左，其余四指握拳，手背朝外。"
请仔细分析文本，识别所有词目和对应的手势操作说明。若无对应手势操作，则操作描述为空。
忽略非手语词目的内容，如页码、标题、注释等。

文本内容:
{chunk}

请严格遵循以下JSON格式返回结果:
[{{"词目": "词目1", "操作": "操作描述1"}}, {{"词目": "词目2", "操作": "操作描述2"}}]
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
            print(f"从当前块中提取了 {len(entries)} 个词条")
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
                print(f"修复后成功解析，提取了 {len(entries)} 个词条")
                all_entries.extend(entries)
            except:
                print("修复JSON后仍然解析失败")
        
        # 添加延迟，避免API请求过于频繁
        time.sleep(2)
        
    # 去重
    unique_entries = []
    seen_entries = set()
    
    for entry in all_entries:
        # 确保键名一致
        if "词目" not in entry and "手语词目" in entry:
            entry["词目"] = entry.pop("手语词目")
        if "操作" not in entry and "手势操作" in entry:
            entry["操作"] = entry.pop("手势操作")
            
        # 创建唯一标识
        key = (entry.get("词目", ""), entry.get("操作", ""))
        if key not in seen_entries and entry.get("词目", "").strip() != "":
            seen_entries.add(key)
            unique_entries.append({"手语词目": entry.get("词目", ""), "手势操作": entry.get("操作", "")})
    
    # 保存最终结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_entries, f, ensure_ascii=False, indent=2)
    print(f"提取结果已保存至: {output_path}")
    
    # 删除临时文件
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    
    return unique_entries

# 处理更小的文本块
def process_smaller_chunk(chunk, index, total, api_url, api_key, all_entries):
    print(f"处理更小的文本块 {index+1}/{total}...")
    
    prompt = f"""
提取以下文本中的手语词目和操作描述，格式为JSON:
{chunk}

例如，当看到类似"脸 lian face 一手五指并拢轻贴一下面颊部"、"G:食指伸直，指尖向左，其余四指握拳，手背朝外。"这样的内容时，应提取为：
"词目": "脸 lian face"
"操作": "一手五指并拢轻贴一下面颊部"
"词目": "G"
"操作": "食指伸直，指尖向左，其余四指握拳，手背朝外。"
请仔细分析文本，识别所有词目和对应的手势操作说明。若无对应手势操作，则操作描述为空。
忽略非手语词目的内容，如页码、标题、注释等。

请返回JSON数组:
"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "deepseek-ai/DeepSeek-V2.5",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 5000
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
                    print(f"从更小的块中提取了 {len(entries)} 个词条")
                    all_entries.extend(entries)
                except json.JSONDecodeError:
                    print("JSON解析错误，尝试修复...")
                    # 尝试修复JSON
                    fixed_json = json_str.replace("'", '"').replace("，", ",")
                    try:
                        entries = json.loads(fixed_json)
                        print(f"修复后成功解析，提取了 {len(entries)} 个词条")
                        all_entries.extend(entries)
                    except:
                        print("修复JSON后仍然解析失败")
        
        # 添加延迟，避免API请求过于频繁
        time.sleep(2)
        
    except Exception as e:
        print(f"处理更小的文本块时出错: {e}")

# 将提取的词条保存为Excel文件
def save_to_excel(entries, output_path):
    if not entries:
        print("没有词条可保存")
        return False
        
    df = pd.DataFrame(entries)
    
    # 确保列名一致
    if "词目" in df.columns and "手语词目" not in df.columns:
        df = df.rename(columns={"词目": "手语词目"})
    if "操作" in df.columns and "手势操作" not in df.columns:
        df = df.rename(columns={"操作": "手势操作"})
    
    # 确保只有需要的列
    if "手语词目" in df.columns and "手势操作" in df.columns:
        df = df[["手语词目", "手势操作"]]
    
    # 过滤掉空词目
    df = df[df["手语词目"].str.strip() != ""]
    
    # 按词目排序
    df = df.sort_values("手语词目")
    
    df.to_excel(output_path, index=False)
    print(f"已将 {len(df)} 个手语词条保存至: {output_path}")
    return True

# 主函数
def main():
    pdf_path = "中国手语.pdf"
    ocr_result_path = "ocr_result.txt"
    extracted_entries_path = "extracted_entries.json"
    output_excel_path = "中国手语词目操作表.xlsx"
    
    # 步骤1: 提取PDF内容或加载已有OCR结果
    if os.path.exists(ocr_result_path):
        print(f"发现已保存的OCR结果: {ocr_result_path}")
        user_input = input("是否使用已保存的OCR结果? (y/n): ").strip().lower()
        if user_input == 'y':
            ocr_text = load_ocr_result(ocr_result_path)
        else:
            print("开始提取PDF内容...")
            ocr_text = extract_text_with_baidu_ocr(pdf_path, start_page=16, save_path=ocr_result_path)
    else:
        print("开始提取PDF内容...")
        ocr_text = extract_text_with_baidu_ocr(pdf_path, start_page=16, save_path=ocr_result_path)
    
    if not ocr_text:
        print("PDF内容提取失败")
        return
    
    # 步骤2: 提取手语词目或加载已有提取结果
    if os.path.exists(extracted_entries_path):
        print(f"发现已保存的提取结果: {extracted_entries_path}")
        user_input = input("是否使用已保存的提取结果? (y/n): ").strip().lower()
        if user_input == 'y':
            with open(extracted_entries_path, 'r', encoding='utf-8') as f:
                entries = json.load(f)
                print(f"已加载提取结果，共 {len(entries)} 个词条")
        else:
            print("开始提取手语词目...")
            entries = extract_sign_language_entries(ocr_text, output_path=extracted_entries_path)
    else:
        print("开始提取手语词目...")
        entries = extract_sign_language_entries(ocr_text, output_path=extracted_entries_path)
    
    # 步骤3: 保存为Excel
    if entries:
        save_to_excel(entries, output_excel_path)
    else:
        print("未能提取到手语词目")

if __name__ == "__main__":
    main() 