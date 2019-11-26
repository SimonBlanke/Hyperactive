import os
import sys
import ntpath

input_path = sys.argv[1]
output_path = sys.argv[2]

file_list = []
for path, subdirs, files in os.walk(input_path):
    for name in files:
        if name.endswith(".py"):
            file_list.append(os.path.join(path, name))

def py2md(py_file_name):
    with open(py_file_name, "r") as py_file:
        py_file_content = py_file.read()

    temp_name, _ = py_file_name.split(".", 2)
    temp_name = ntpath.basename(temp_name)
    md_file_name = output_path + "/" + temp_name + ".md"

    front_str = "```python\n"
    back_str = "```"
    md_file_content = front_str + py_file_content + back_str

    md_file = open(md_file_name, "w")
    md_file.write(md_file_content)
    md_file.close()
    print("Wrote", py_file_name, "--->", md_file_name)

for i_arg in range(len(file_list)):
    py_file_name = file_list[i_arg]
    py2md(py_file_name)
