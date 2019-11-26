import sys

def py2md(py_file_name):
    with open(py_file_name, "r") as py_file:
        py_file_content = py_file.read()

    temp_name, _ = py_file_name.split(".", 2)
    md_file_name = temp_name + ".md"

    front_str = "```python\n"
    back_str = "```"
    md_file_content = front_str + py_file_content + back_str

    md_file = open(md_file_name, "w")
    md_file.write(md_file_content)
    md_file.close()

for i_arg in range(1, len(sys.argv)):
    py_file_name = sys.argv[i_arg]
    py2md(py_file_name)
