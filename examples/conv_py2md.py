import glob
import sys
import ntpath

input_path = sys.argv[1]
output_path = sys.argv[2]

py_file_list = glob.glob(input_path + "/*.py")


def py2md(py_file_list):
    md_total_content = ""

    for py_file_name in py_file_list:
        with open(py_file_name, "r") as py_file:
            py_file_content = py_file.read()

        temp_name, _ = py_file_name.split(".", 2)
        temp_name = ntpath.basename(temp_name)

        front_str = "```python\n"
        back_str = "```"
        md_file_content = front_str + py_file_content + back_str

        md_total_content = (
            md_total_content + "## " + temp_name + "\n\n" + md_file_content + "\n\n"
        )

    md_file_name = output_path + "/" + input_path + ".md"
    md_file = open(md_file_name, "w+")
    md_file.write(md_total_content)
    md_file.close()
    print("Wrote", py_file_list, "--->", md_file_name)


py2md(py_file_list)
