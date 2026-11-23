"""Bank Marketing Analytics dashboard and native multi-page navigation."""

import streamlit as st

import config
from utils.dashboard import (
    CsvTable,
    apply_theme,
    load_csv,
    load_json,
    load_svg,
    render_kpi_card,
    render_navigation_card,
    render_page_header,
    render_section_heading,
    render_sidebar_brand,
    render_svg,
)


st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)
apply_theme()
render_sidebar_brand()


def _target_counts(
    dataset: CsvTable | None,
    target_distribution: CsvTable | None,
) -> dict[str, int]:
    counts = {"no": 0, "yes": 0}
    if dataset is not None and "y" in dataset.columns:
        for row in dataset.rows:
            label = row.get("y", "").strip().casefold()
            if label in counts:
                counts[label] += 1
        return counts

    if target_distribution is not None:
        for row in target_distribution.rows:
            label = row.get("y", "").strip().casefold()
            if label in counts:
                try:
                    counts[label] = int(row["records"])
                except (KeyError, ValueError):
                    counts[label] = 0
    return counts


def render_dashboard() -> None:
    render_page_header(
        "Welcome to Bank Marketing Analytics",
        "Explore campaign outcomes, review model performance, and navigate the "
        "subscription prediction workflow. All dashboard metrics are sourced "
            "from the original dataset and the current duplicate-safe v2 experiment.",
        eyebrow="Bank marketing campaign analytics",
    )

    dataset = load_csv("data/raw/01_original_dataset.csv", "Original dataset")
    target_distribution = load_csv(
        "reports/dataset_analysis/target_distribution.csv",
        "Target distribution",
    )
    cv_comparison = load_csv(
        "reports/model_training/duplicate_safe_v2/model_comparison_summary.csv",
        "duplicate_safe_v2 cross-validation comparison",
    )
    test_metrics = load_json(
        "reports/model_evaluation/duplicate_safe_v2/test_metrics.json",
        "duplicate_safe_v2 final test metrics",
    )

    class_counts = _target_counts(dataset, target_distribution)
    if dataset is not None:
        record_count = len(dataset.rows)
        input_features = (
            len(dataset.columns) - 1
            if "y" in dataset.columns
            else None
        )
    else:
        record_count = None
        input_features = None

    source_has_target = dataset is not None and "y" in dataset.columns
    total_target_count = (
        record_count
        if source_has_target and record_count is not None
        else sum(class_counts.values())
    )
    target_available = total_target_count > 0 and sum(class_counts.values()) > 0
    subscription_rate = (
        class_counts["yes"] / total_target_count * 100
        if target_available
        else None
    )
    evaluated_algorithm_count = len(cv_comparison.rows) if cv_comparison else 0

    render_section_heading("Campaign at a glance")
    kpi_columns = st.columns(4)
    with kpi_columns[0]:
        render_kpi_card(
            "Campaign records",
            f"{record_count:,}" if record_count is not None else "Unavailable",
            context="Original source CSV · read-only",
        )
    with kpi_columns[1]:
        feature_value = (
            f"{input_features} + 1"
            if input_features is not None and source_has_target
            else "Unavailable"
        )
        render_kpi_card(
            "Features + target",
            feature_value,
            context="Input attributes + target y",
            help_text="The source CSV contains input attributes and a separate y target column.",
        )
    with kpi_columns[2]:
        render_kpi_card(
            "Subscription rate",
            f"{subscription_rate:.2f}%" if subscription_rate is not None else "Unavailable",
            context=(
                f"{class_counts['yes']:,} positive labels"
                if subscription_rate is not None
                else "Target counts are unavailable"
            ),
        )
    with kpi_columns[3]:
        render_kpi_card(
            "v2 algorithms evaluated",
            str(evaluated_algorithm_count),
            context="Current duplicate-safe experiment",
        )

    st.divider()
    overview_columns = st.columns((1.05, 0.95), gap="large")
    with overview_columns[0]:
        with st.container(border=True):
            render_section_heading(
                "Subscription overview",
                "The target distribution from the original campaign dataset.",
            )
            target_columns = st.columns(2)
            target_columns[0].metric(
                "Subscribed · yes",
                f"{class_counts['yes']:,}" if target_available else "Unavailable",
            )
            target_columns[1].metric(
                "Did not subscribe · no",
                f"{class_counts['no']:,}" if target_available else "Unavailable",
            )
            target_chart = load_svg(
                "reports/eda/target_distribution.svg",
                "Target distribution chart",
            )
            if target_chart is not None:
                render_svg(target_chart, "Subscription class distribution")
            elif target_distribution is None:
                st.info("The target chart is not available.")
    with overview_columns[1]:
        with st.container(border=True):
            render_section_heading(
                "Current experiment: duplicate_safe_v2",
                "Random Forest selected from six models using training-partition cross-validation.",
            )
            st.markdown("**Selected model: Random Forest**")
            random_forest_cv = next(
                (
                    row
                    for row in (cv_comparison.rows if cv_comparison else [])
                    if row.get("model") == "random_forest"
                ),
                None,
            )
            cv_column, test_f1_column = st.columns(2)
            with cv_column:
                st.metric(
                    "Mean CV F1 (training partition)",
                    f"{float(random_forest_cv['mean_f1']):.4f}"
                    if random_forest_cv
                    else "Unavailable",
                )
            with test_f1_column:
                st.metric(
                    "Final held-out test F1",
                    f"{float(test_metrics['f1_score']):.4f}"
                    if test_metrics
                    else "Unavailable",
                )
            test_recall_column, test_auc_column = st.columns(2)
            with test_recall_column:
                st.metric(
                    "Final held-out test recall",
                    f"{float(test_metrics['recall']):.4f}"
                    if test_metrics
                    else "Unavailable",
                )
            with test_auc_column:
                st.metric(
                    "Final held-out test ROC-AUC",
                    f"{float(test_metrics['roc_auc']):.4f}"
                    if test_metrics
                    else "Unavailable",
                )
            st.caption(
                "CV and final test metrics come from separate v2 partitions. The "
                "held-out test set has already been used and must not be reused for tuning."
            )

    render_section_heading(
        "duplicate_safe_v2 cross-validation comparison",
        "All six algorithms on the v2 training partition. These are CV results, not test scores.",
    )
    if cv_comparison is None:
        st.info("The v2 cross-validation comparison is unavailable.")
    else:
        overview_rows = []
        for row in cv_comparison.rows:
            try:
                overview_rows.append(
                    {
                        "Model": row["display_name"],
                        "Mean CV F1": f"{float(row['mean_f1']):.4f}",
                    }
                )
            except (KeyError, ValueError):
                continue
        st.dataframe(overview_rows, hide_index=True, width="stretch")

    st.divider()
    render_section_heading(
        "Explore the project",
        "Open a completed analysis page or use the existing prediction workflow.",
    )
    navigation_cards = (
        (
            "Dataset Explorer",
            "Browse the original data, summaries, and target distribution.",
            "pages/1_Dataset_Explorer.py",
            ":material/table_view:",
        ),
        (
            "EDA & Analytics",
            "Review the five existing exploratory analysis charts and findings.",
            "pages/2_EDA_Analytics.py",
            ":material/insights:",
        ),
        (
            "Model Development",
            "See the six duplicate-safe v2 models and training-partition CV results.",
            "pages/4_Model_Development.py",
            ":material/model_training:",
        ),
        (
            "Model Evaluation",
            "Review the final held-out v2 test metrics, confusion matrix, and curves.",
            "pages/6_Model_Evaluation.py",
            ":material/assessment:",
        ),
        (
            "Customer Prediction",
            "Use the selected v2 Random Forest for one validated customer record.",
            "pages/7_Customer_Prediction.py",
            ":material/person_search:",
        ),
        (
            "Bulk Prediction",
            "Validate a CSV and download predictions with row-level errors.",
            "pages/8_Bulk_Prediction.py",
            ":material/upload_file:",
        ),
    )
    navigation_columns = st.columns(3, gap="medium")
    for index, (title, description, page_path, icon) in enumerate(navigation_cards):
        with navigation_columns[index % len(navigation_columns)]:
            render_navigation_card(title, description, page_path, icon)

    st.info(
        "Model Development shows v2 group-aware cross-validation on the training "
        "partition. Model Evaluation shows the separate, one-time held-out v2 test results."
    )


page_groups = {
    "Overview": [
        st.Page(
            render_dashboard,
            title="Dashboard",
            icon=":material/dashboard:",
            default=True,
        ),
    ],
    "Data & insights": [
        st.Page(
            "pages/1_Dataset_Explorer.py",
            title="Dataset Explorer",
            icon=":material/table_view:",
        ),
        st.Page(
            "pages/2_EDA_Analytics.py",
            title="EDA & Analytics",
            icon=":material/insights:",
        ),
        st.Page(
            "pages/3_Preprocessing.py",
            title="Preprocessing",
            icon=":material/tune:",
        ),
    ],
    "Modeling": [
        st.Page(
            "pages/4_Model_Development.py",
            title="Model Development",
            icon=":material/model_training:",
        ),
        st.Page(
            "pages/5_Train_Test_Split_CV.py",
            title="Train/Test Split & Cross-Validation",
            icon=":material/splitscreen:",
        ),
        st.Page(
            "pages/6_Model_Evaluation.py",
            title="Model Evaluation",
            icon=":material/assessment:",
        ),
    ],
    "Predictions": [
        st.Page(
            "pages/7_Customer_Prediction.py",
            title="Customer Prediction",
            icon=":material/person_search:",
        ),
        st.Page(
            "pages/8_Bulk_Prediction.py",
            title="Bulk Prediction",
            icon=":material/upload_file:",
        ),
    ],
    "Guidance": [
        st.Page(
            "pages/9_Ethical_Considerations.py",
            title="Ethical Considerations",
            icon=":material/verified_user:",
        ),
    ],
}
navigation = st.navigation(page_groups)
navigation.run()
