# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

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

    for search_id in search_ids:
        st.title(search_id)
        st.components.v1.html(
            """<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """,
            height=10,
        )
        col1, col2 = st.beta_columns([1, 2])

        pyplot_fig, plotly_fig = backend.create_plots(search_id)

        if pyplot_fig is not None:
            col1.pyplot(pyplot_fig)
        if plotly_fig is not None:
            col2.plotly_chart(plotly_fig)

        for _ in range(3):
            st.write(" ")

    time.sleep(1)
    print("\nStart next run:")
    st.experimental_rerun()


if __name__ == "__main__":
    main()
