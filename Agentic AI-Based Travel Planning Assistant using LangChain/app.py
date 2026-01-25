import streamlit as st
import pandas as pd
from agent import run_agent
from pdf_generator import generate_pdf

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
}
.sub-title {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 25px;
}
.card {
    background-color: #f9fafb;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 18px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">✈️ Agentic AI Travel Planning Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced offline AI-inspired travel planning system</div>', unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.header("🧭 Trip Preferences")

max_budget = st.sidebar.slider("💰 Max Budget (₹)", 5000, 100000, 25000, step=1000)
min_stars = st.sidebar.slider("⭐ Minimum Hotel Stars", 1, 5, 3)
days_per_destination = st.sidebar.slider("🗓️ Days per Destination", 1, 7, 3)

destinations_input = st.sidebar.text_input(
    "📍 Destinations (comma separated)",
    "Goa, Mumbai"
)

use_llm = st.sidebar.toggle("🤖 Use LLM (Future Feature)", value=False)
st.sidebar.caption("LLM toggle is disabled in offline mode but kept for future integration.")

destinations = [d.strip().title() for d in destinations_input.split(",") if d.strip()]

query = st.text_input(
    "Describe your trip",
    "Plan a multi destination trip with budget optimization"
)

# ================= GENERATE PLAN =================
if st.button("🚀 Generate Travel Plan"):
    with st.spinner("Planning your trip like a travel expert..."):
        st.session_state.result = run_agent(
            query=query,
            max_budget=max_budget,
            min_stars=min_stars,
            destinations=destinations,
            days=days_per_destination
        )
    st.session_state.history.append(st.session_state.result)

# ================= DISPLAY RESULT =================
if st.session_state.result:
    result = st.session_state.result

    if not result["Trips"]:
        st.warning("No destinations matched your filters. Try increasing budget or lowering hotel stars.")
        st.stop()

    # ===== SUMMARY METRICS =====
    total_cost = sum(t["Budget Breakdown"]["Total"] for t in result["Trips"])
    total_days = len(result["Trips"]) * days_per_destination

    c1, c2, c3 = st.columns(3)
    c1.metric("📍 Destinations", len(result["Trips"]))
    c2.metric("🗓️ Total Days", total_days)
    c3.metric("💰 Estimated Cost", f"₹{total_cost}")

    st.divider()
    st.subheader(result["Trip Summary"])

    # ===== TRIP DETAILS =====
    for trip in result["Trips"]:
        st.markdown(f"## 📍 {trip['Destination']}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### ✈️ Flight")
            st.write(f"Airline: {trip['Flight Selected']['airline']}")
            st.write(f"Route: {trip['Flight Selected']['from']} → {trip['Flight Selected']['to']}")
            st.write(f"Departure: {trip['Flight Selected']['departure_time']}")
            st.write(f"Arrival: {trip['Flight Selected']['arrival_time']}")
            st.write(f"Price: ₹{trip['Flight Selected']['price']}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🏨 Hotel")
            st.write(f"Name: {trip['Hotel Selected']['name']}")
            st.write(f"Stars: ⭐ {trip['Hotel Selected']['stars']}")
            st.write(f"Price/Night: ₹{trip['Hotel Selected']['price_per_night']}")
            st.write(f"Amenities: {', '.join(trip['Hotel Selected']['amenities'])}")
            st.markdown('</div>', unsafe_allow_html=True)

        # ===== ITINERARY =====
        st.markdown("### 📅 Day-wise Itinerary")
        for day, places in trip["Day-wise Itinerary"].items():
            st.markdown(f"**{day}**")
            for p in places:
                st.markdown(f"- {p}")

        # ===== WEATHER =====
        st.markdown("### ☀️ Weather Forecast")
        for day, weather in trip["Weather Forecast"].items():
            st.markdown(f"- {day}: {weather}")

        # ===== BUDGET =====
        budget = trip["Budget Breakdown"]
        st.markdown("### 💰 Budget Breakdown")
        st.markdown(f"""
        - Flight: ₹{budget['Flight']}
        - Hotel: ₹{budget['Hotel']}
        - Food & Local Travel: ₹{budget['Food & Local Travel']}
        **Total: ₹{budget['Total']}**
        """)

        # ===== COST CHART =====
        st.markdown("### 📊 Cost Breakdown")
        df = pd.DataFrame({
            "Category": ["Flight", "Hotel", "Food & Local"],
            "Cost": [
                budget["Flight"],
                budget["Hotel"],
                budget["Food & Local Travel"]
            ]
        }).set_index("Category")

        st.bar_chart(df)

        st.divider()

    # ===== REASONING =====
    with st.expander("🤖 Why this plan?"):
        for reason in result["Agent Reasoning"]:
            st.write(f"- {reason}")

    # ===== PDF DOWNLOAD =====
    pdf = generate_pdf(result)
    st.download_button(
        "📄 Download Itinerary as PDF",
        pdf,
        "travel_itinerary.pdf",
        "application/pdf"
    )

# ================= COMPARE TWO PLANS =================
st.markdown("## 🔍 Compare Two Plans")

if len(st.session_state.history) >= 2:
    a, b = st.columns(2)

    with a:
        p1 = st.selectbox(
            "Plan A",
            range(len(st.session_state.history)),
            format_func=lambda x: st.session_state.history[x]["Trip Summary"]
        )

    with b:
        p2 = st.selectbox(
            "Plan B",
            range(len(st.session_state.history)),
            format_func=lambda x: st.session_state.history[x]["Trip Summary"]
        )

    if st.button("📊 Compare"):
        def stats(plan):
            cost = sum(t["Budget Breakdown"]["Total"] for t in plan["Trips"])
            days = sum(len(t["Day-wise Itinerary"]) for t in plan["Trips"])
            dest = len(plan["Trips"])
            return cost, days, dest

        c1, d1, s1 = stats(st.session_state.history[p1])
        c2, d2, s2 = stats(st.session_state.history[p2])

        x, y, z = st.columns(3)
        x.metric("💰 Cost Diff", f"₹{c1}", f"₹{c1 - c2}")
        y.metric("🗓️ Days Diff", d1, d1 - d2)
        z.metric("📍 Destinations Diff", s1, s1 - s2)
else:
    st.info("Generate at least two plans to enable comparison.")
