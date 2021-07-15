# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import sys
import time
import streamlit as st


from streamlit_backend import StreamlitBackend


def main():
    try:
        st.set_page_config(page_title="Hyperactive Progress Board", layout="wide")
    except:
        pass

    search_ids = sys.argv[1:]
    backend = StreamlitBackend(search_ids)
    lock_files = []

    for search_id in search_ids:
        st.title(search_id)
        st.components.v1.html(
            """<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """,
            height=10,
        )
        st.write(" ")

        _, col_2, _, col_4 = st.beta_columns([0.1, 0.9, 0.2, 1.8])
        col1, col2 = st.beta_columns([1, 2])

        progress_data = backend.get_progress_data(search_id)

        pyplot_fig = backend.pyplot(progress_data)
        plotly_fig = backend.plotly(progress_data, search_id)

        if pyplot_fig is not None:
            col_2.header("Best score progression")
            col1.pyplot(pyplot_fig)
        if plotly_fig is not None:
            col_4.header("Parallel Coordinates")
            col2.plotly_chart(plotly_fig)

        last_best = backend.create_info(search_id)

        _, col2 = st.beta_columns([0.1, 0.9])
        if last_best is not None:
            last_best = last_best.assign(hack="").set_index("hack")
            st.write(" ")
            col2.header("Up to 5 best scores information")
            st.table(last_best)

        for _ in range(3):
            st.write(" ")

        lock_file = backend._io_.get_lock_file_path(search_id)
        lock_files.append(os.path.isfile(lock_file))

    time.sleep(1)
    if all(lock_file is False for lock_file in lock_files):
        print("\n --- Deleting progress- and filter-files --- \n")

        for search_id in search_ids:
            backend._io_.remove_progress(search_id)
            backend._io_.remove_filter(search_id)

    else:
        print("\n --- Rerun streamlit --- \n")
        st.experimental_rerun()


if __name__ == "__main__":
    main()
