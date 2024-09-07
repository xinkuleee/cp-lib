import os

# 获取当前目录
current_dir = os.getcwd()

for entry1 in os.scandir(current_dir):
    if entry1.is_dir():
        if entry1.name in ["Basic", "Data", "DP", "Geometry", "Graph", "Math", "String", "Template"]:
            files = []
            for entry2 in os.scandir(entry1):
                if entry2.is_file():
                    if entry2.name.endswith(".cpp") or entry2.name.endswith(".py") or entry2.name in [".clang-format", "Makefile", "settings.json"]:
                        files.append(entry2.name)
			# 将文件名写入 main.tex 文件
            with open("./" + entry1.name + "/" + "main.tex", "w", encoding="utf-8") as f:  # 添加 encoding="utf-8"
                f.write("\\section{" + entry1.name + "}\n\n")
                for file in files:
                    if file == 'ref.py':
                        continue
                    title = file.replace("&", " ").replace("_", "-")
                    f.write("\\subsection{" + title + "}\n")
                    f.write("\\include{" + entry1.name + "/" + file + "}\n\n")