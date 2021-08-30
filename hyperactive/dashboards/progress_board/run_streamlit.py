# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import sys
import time
import streamlit as st


from streamlit_backend import StreamlitBackend


sys.setrecursionlimit(10000)


def main():
    try:
        st.set_page_config(page_title="Hyperactive Progress Board", layout="wide")
    except:
        pass

    progress_ids = sys.argv[1:]
    backend = StreamlitBackend(progress_ids)
    lock_files = []

    for progress_id in progress_ids:
        search_id = progress_id.rsplit(":")[0]

        st.title(search_id)
        st.components.v1.html(
            """<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """,
            height=10,
        )
        st.write(" ")

        _, col_2, _, col_4 = st.beta_columns([0.1, 0.9, 0.1, 2])
        col1, col2 = st.beta_columns([1, 2])

        progress_data = backend.get_progress_data(progress_id)

        pyplot_fig = backend.pyplot(progress_data)
        plotly_fig = backend.plotly(progress_data, progress_id)

        if pyplot_fig is not None:
            col_2.header("Best score progression")
            col1.pyplot(pyplot_fig)
        if plotly_fig is not None:
            col_4.header("Parallel Coordinates")
            col2.plotly_chart(plotly_fig, use_container_width=True)

        last_best = backend.create_info(progress_id)

        if last_best is not None:
            plotly_table = backend.table_plotly(last_best)
            st.plotly_chart(plotly_table, use_container_width=True)

        for _ in range(3):
            st.write(" ")

        lock_file = backend._io_.get_lock_file_path(progress_id)
        lock_files.append(os.path.isfile(lock_file))

    time.sleep(3)
    if all(lock_file is False for lock_file in lock_files):
        print("\n --- Deleting progress- and filter-files --- \n")

        for progress_id in progress_ids:
            backend._io_.remove_progress(progress_id)
            backend._io_.remove_filter(progress_id)

    else:
        print("\n --- Rerun streamlit --- \n")
        st.experimental_rerun()


if __name__ == "__main__":
    main()
