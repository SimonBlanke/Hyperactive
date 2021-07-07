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
        col1, col2 = st.beta_columns([1, 2])

        progress_data = backend.get_progress_data(search_id)

        pyplot_fig = backend.pyplot(progress_data)
        plotly_fig = backend.plotly(progress_data, search_id)

        if pyplot_fig is not None:
            col1.pyplot(pyplot_fig)
        if plotly_fig is not None:
            col2.plotly_chart(plotly_fig)

        last_best = backend.create_info(search_id)
        if last_best is not None:
            last_best = last_best.assign(hack="").set_index("hack")
            st.table(last_best)

        for _ in range(3):
            st.write(" ")

        lock_file = backend._io_.get_lock_file_path(search_id)
        lock_files.append(os.path.isfile(lock_file))

    print("\n lock_files", lock_files, "\n")

    time.sleep(1)
    if all(lock_file is False for lock_file in lock_files):
        print("\n --- Deleting progress-/filter-files ---")

        for search_id in search_ids:
            backend._io_.remove_progress(search_id)
            backend._io_.remove_filter(search_id)

    else:
        print("\n --- Rerun streamlit ---")
        st.experimental_rerun()


if __name__ == "__main__":
    main()
