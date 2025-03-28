import os

"""
用于读取程序代码生成提示词的脚本
使用方法：
1. 修改 BASE_DIR 为你的目标目录路径
2. 修改 INTRODUCTION_TEXT 为你想要的给AI的提示词
3. 运行脚本，生成的提示词将保存在 OUTPUT_FILE 中
4. 将 OUTPUT_FILE 中的内容复制到你的AI模型中，让AI根据提示词生成代码
"""


def main():
    # 用户配置区域 ================================================
    BASE_DIR = "../DQN_from_zly"  # 请修改为你的目标目录路径
    INTRODUCTION_TEXT = """ \
你是一个专业程序员，请你阅读下面的python文件，给我讲解这些文件分别是用来做什么的

我已经将这些文件的文件名标注在内容的前面。
"""
    OUTPUT_FILE = "prompt.txt"  # 生成的输出文件名
    FILE_EXTENSIONS = ('.py', '.txt')  # 要处理的文件扩展名
    # ============================================================

    print("开始文件扫描...")
    file_paths = find_files(BASE_DIR, FILE_EXTENSIONS)
    print(f"找到 {len(file_paths)} 个可处理文件")

    print("正在读取文件内容...")
    files_content = read_files(file_paths, BASE_DIR)

    print("生成最终文档...")
    full_content = INTRODUCTION_TEXT + "\n\n" + files_content

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_content)
    print(f"生成完成！输出文件已保存至：{os.path.abspath(OUTPUT_FILE)}")


def find_files(directory, extensions):
    """递归查找目录下指定扩展名的文件"""
    file_list = []
    target_ext = {ext.lower() for ext in extensions}

    for root, _, files in os.walk(directory):
        for filename in files:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in target_ext:
                full_path = os.path.join(root, filename)
                file_list.append(full_path)
    return file_list


def read_files(file_paths, base_dir):
    """读取文件内容并格式化"""
    contents = []
    for path in file_paths:
        # 生成相对路径作为标题
        rel_path = os.path.relpath(path, base_dir)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                # 使用Markdown语法突出显示文件名
                contents.append(f"## File: {rel_path}\n```\n{f.read()}\n```\n")
        except UnicodeDecodeError:
            print(f"解码失败跳过文件：{rel_path}（可能包含二进制内容）")
        except Exception as e:
            print(f"读取文件异常 {rel_path}: {str(e)}")
    return '\n'.join(contents)


if __name__ == "__main__":
    main()
