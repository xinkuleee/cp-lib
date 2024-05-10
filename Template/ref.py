import os

def get_cpp_and_python_files(directory):
    """
    获取指定目录下所有 .cpp 和 .python 文件的名称

    Args:
        directory: 目标目录

    Returns:
        一个包含所有 .cpp 和 .python 文件名称的列表
    """
    files = []
    for entry in os.scandir(directory):
        if entry.is_file():
            files.append(entry.name)
        elif entry.is_dir():
            files.extend(get_cpp_and_python_files(entry.path))  # 递归处理子文件夹
    return files

# 获取当前目录
current_dir = os.getcwd()

# 获取当前目录的名称
current_dir_name = os.path.basename(current_dir)

# 获取所有 .cpp 和 .python 文件
files = get_cpp_and_python_files(current_dir)

# 将文件名写入 main.tex 文件
with open("main.tex", "w", encoding="utf-8") as f:  # 添加 encoding="utf-8"
    f.write("\\section{" + current_dir_name + "}\n\n")
    for file in files:
        if file == 'ref.py' or file == 'todo.txt':
            continue
        f.write("\\subsection{" + file + "}\n")
        f.write("\\include{" + current_dir_name + "/" + file + "}\n\n")
