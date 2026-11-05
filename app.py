import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# 1. Page Configuration & Custom Styling
st.set_page_config(
    page_title="Go Ananas! - AI Plant Doctor",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        padding: 0.6rem;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 2. Robust Model Loading (Bypasses metadata errors & maps weights safely)
@st.cache_resource
def load_pineapple_model():
    try:
        # Re-create MobileNetV2 architecture with explicitly named layers
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), 
            include_top=False, 
            weights=None
        )
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        outputs = tf.keras.layers.Dense(6, activation="softmax", name='output_dense')(x)
        
        model = tf.keras.Model(inputs, outputs, name='pineapple_mobilenetv2')
        
        # Safely map matching weights while skipping incompatible top layers or metadata
        model.load_weights("go_ananas_model.h5", by_name=True, skip_mismatch=True)
        return model
        
    except Exception as e:
        st.error(f"Detailed Model Load Error: {e}")
        return None


model = load_pineapple_model()

# Alphabetical order matching 6-class directory structure
class_labels = [
    "Fruit Rot",            # Index 0
    "Healthy",              # Index 1
    "Heart Rot",            # Index 2
    "Mealybug Wilt",        # Index 3
    "Multi-Crown Disorder", # Index 4
    "Root Rot",             # Index 5
]

# Actionable management advice
recommendations = {
    "Healthy": (
        "✅ **Crop Status: Optimal.** Continue standard irrigation and "
        "balanced fertilization schedules. No immediate action required."
    ),
    "Fruit Rot": (
        "⚠️ **Action Required:** Remove infected fruits immediately to prevent "
        "fungal spore dispersal. Apply approved bio-fungicides and avoid "
        "overhead irrigation."
    ),
    "Heart Rot": (
        "⚠️ **Action Required:** Remove and destroy infected heart/whorl tissue. "
        "Apply copper-based fungicides and improve field drainage to prevent bacterial spread."
    ),
    "Mealybug Wilt": (
        "⚠️ **Action Required:** Quarantine affected plants. Apply systemic "
        "insecticides or introduce natural predators (e.g., ladybugs) to control "
        "mealybug populations."
    ),
    "Multi-Crown Disorder": (
        "⚠️ **Action Required:** Monitor Calcium and Zinc: Deficiencies in "
        "essential micronutrients have been linked to structural "
        "malformations. Conduct routine soil and foliar tests to ensure proper "
        "nutrient balance."
    ),
    "Root Rot": (
        "⚠️ **Action Required:** Improve field drainage immediately. Avoid "
        "waterlogging, reduce watering frequency, and treat soil with "
        "appropriate fungicides."
    ),
}

# 3. Sidebar Configuration
with st.sidebar:
    st.title("Go Ananas! 🍍")
    st.markdown("### AI-Powered Pineapple Disease Detection")
    st.markdown("---")
    st.info(
        "Upload a photo of a pineapple leaf, crown, or fruit section to diagnose "
        "health conditions instantly."
    )
    st.markdown("**Supported Conditions:**")
    st.markdown(
        "- 🔴 Fruit Rot\n"
        "- 🟢 Healthy\n"
        "- 🤎 Heart Rot\n"
        "- 🟡 Mealybug Wilt\n"
        "- 👑 Multi-Crown Disorder\n"
        "- 🟤 Root Rot"
    )
    st.markdown("---")
    show_debug = st.checkbox("Show Raw Output Array", value=False)

# 4. Main Interface Header
st.title("🍍 Go Ananas! Disease Diagnostic Dashboard")
st.markdown(
    "Welcome to your smart agricultural assistant. Upload an image below to run "
    "pattern recognition."
)
st.markdown("---")

# File Uploader
uploaded_file = st.file_uploader(
    "Choose a pineapple image (.jpg, .png, .jpeg)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📸 Uploaded Specimen")
        image_pil = Image.open(uploaded_file)
        st.image(
            image_pil, caption="Target Specimen for Analysis", use_container_width=True
        )

    with col2:
        st.subheader("🔍 Diagnostic Results")

        with st.spinner("Analyzing plant tissue patterns..."):
            img_resized = image_pil.resize((224, 224))

            if img_resized.mode != "RGB":
                img_resized = img_resized.convert("RGB")

            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            if model is not None:
                predictions = model.predict(img_array)[0]
            else:
                st.error("Model could not be loaded. Please check model file.")
                predictions = np.zeros(len(class_labels))

            predicted_idx = int(np.argmax(predictions))
            predicted_class = class_labels[predicted_idx]
            confidence_score = float(predictions[predicted_idx]) * 100

        if predicted_class == "Healthy":
            st.success(
                f"**Diagnosis:** {predicted_class} ({confidence_score:.2f}% Confidence)"
            )
        else:
            st.error(
                f"**Diagnosis:** {predicted_class} ({confidence_score:.2f}% Confidence)"
            )

        st.markdown("#### 💡 Recommended Action")
        st.markdown(recommendations[predicted_class])

        st.markdown("---")
        st.subheader("📊 Confidence Breakdown Across All Classes")

        for idx, label in enumerate(class_labels):
            prob = float(predictions[idx])
            st.text(f"{label}: {prob * 100:.1f}%")
            st.progress(prob)

        if show_debug:
            st.markdown("---")
            st.write("Raw Model Probabilities:", predictions)

else:
    st.info(
        "👈 Upload an image using the box above or via the sidebar to begin analysis."
    )
