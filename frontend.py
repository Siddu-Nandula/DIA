
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DIA (Document Intelligence Agent)",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DIA (Document Intelligence Agent)")
st.caption(
    "Analyze documents, ask questions, discover insights, "
    "and generate charts using your local AI system."
)

# ---------------------------------------------------------
# SIDEBAR: SYSTEM STATUS
# ---------------------------------------------------------

with st.sidebar:
    st.header("System Status")

    try:
        health_response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )
        health_response.raise_for_status()
        health = health_response.json()

        st.success("API is responding")
        st.write(f"**Provider:** {health.get('llm_provider', 'Unknown')}")
        st.write(f"**Model:** {health.get('llm_model', 'Unknown')}")

    except requests.RequestException:
        st.error("API unavailable")
        st.caption("Start FastAPI at http://127.0.0.1:8000")

    st.divider()
    st.markdown("**Supported files**")
    st.write("PDF, CSV, XLSX")


# ---------------------------------------------------------
# DOCUMENT UPLOAD
# ---------------------------------------------------------

st.subheader("1. Upload your document")

uploaded_file = st.file_uploader(
    "Choose a PDF, CSV, or Excel file",
    type=["pdf", "csv", "xlsx"],
)


# ---------------------------------------------------------
# USER QUESTION
# ---------------------------------------------------------

st.subheader("2. Ask a question")

question = st.text_area(
    "What would you like to know?",
    placeholder=(
        "Examples:\n"
        "- Summarize this document\n"
        "- What is the total revenue?\n"
        "- Which region generated the highest revenue?\n"
        "- Create a chart of revenue by region\n"
        "- Show monthly revenue trends\n"
        "- Plot units sold over time\n"
        "- Generate a histogram of units"
    ),
    height=140,
)


# ---------------------------------------------------------
# ANALYZE DOCUMENT
# ---------------------------------------------------------

if st.button(
    "Analyze Document",
    type="primary",
    use_container_width=True,
):

    if uploaded_file is None:
        st.warning("Please upload a document first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Analyzing your document..."):

            try:
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    },
                    data={"question": question.strip()},
                    timeout=600,
                )

                if response.ok:
                    result = response.json()

                    st.session_state["last_answer"] = result.get(
                        "answer",
                        "No answer returned.",
                    )

                    st.session_state["last_tool"] = result.get(
                        "tool_used",
                        "unknown",
                    )

                    st.session_state["last_question"] = question.strip()

                else:
                    try:
                        detail = response.json().get(
                            "detail",
                            response.text,
                        )
                    except ValueError:
                        detail = response.text

                    st.error(
                        f"Request failed ({response.status_code}): {detail}"
                    )

            except requests.Timeout:
                st.error(
                    "The request timed out. Your local model may need "
                    "more time; try again."
                )

            except requests.RequestException as exc:
                st.error(f"Could not connect to the API: {exc}")


# ---------------------------------------------------------
# DISPLAY ANALYSIS RESULT
# ---------------------------------------------------------

if "last_answer" in st.session_state:

    st.divider()
    st.subheader("3. Analysis Result")

    st.markdown("**Your question**")
    st.write(st.session_state["last_question"])

    st.markdown("**Answer**")
    st.markdown(st.session_state["last_answer"])

    st.caption(
        f"Tool used: {st.session_state['last_tool']}"
    )

    st.download_button(
        "Download answer as text",
        data=st.session_state["last_answer"],
        file_name="document_analysis.txt",
        mime="text/plain",
    )


    # -----------------------------------------------------
    # CHART DISPLAY
    # -----------------------------------------------------

    chart_names = [
        "agent_chart.png",
        "agent_histogram.png",
        "agent_line_chart.png",
    ]

    chart_keywords = (
        "chart",
        "charts",
        "graph",
        "plot",
        "histogram",
        "visualize",
        "visualise",
        "distribution",
        "trend",
        "trends",
        "over time",
        "time series",
        "line chart",
        "line graph",
        "monthly",
        "daily",
        "weekly",
        "yearly",
        "by date",
        "by month",
        "by day",
        "by year",
        "month by month",
        "day by day",
        "week by week",
        "year by year",
    )

    last_question = st.session_state["last_question"].lower()

    asks_for_chart = any(
        keyword in last_question
        for keyword in chart_keywords
    )

    if asks_for_chart:

        chart_images = []

        for chart_name in chart_names:

            try:
                chart_response = requests.get(
                    f"{API_BASE_URL}/charts/{chart_name}",
                    timeout=10,
                )

                if chart_response.status_code == 200:
                    chart_images.append(
                        (chart_name, chart_response.content)
                    )

            except requests.RequestException:
                st.warning(
                    "Could not retrieve charts from the API."
                )

        if chart_images:

            st.divider()
            st.subheader("4. Generated Charts")

            for chart_name, chart_content in chart_images:

                st.image(
                    chart_content,
                    caption=(
                        chart_name
                        .replace("_", " ")
                        .replace(".png", "")
                        .title()
                    ),
                    use_container_width=True,
                )

                st.download_button(
                    f"Download {chart_name}",
                    data=chart_content,
                    file_name=chart_name,
                    mime="image/png",
                    key=f"download_{chart_name}",
                )

        else:
            st.info(
                "No generated chart was found. "
                "Check whether the API chart endpoint is serving "
                "the generated image."
            )
