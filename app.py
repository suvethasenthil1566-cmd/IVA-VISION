# NEXUS VISION — single-file Streamlit application
# Existing page layouts and result outputs are preserved from the supplied files.

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import tempfile
import os
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine


def template_matching_page():

    st.markdown(
        """
        <h1 style="text-align:center; color:#38bdf8;">
        🔍 OBJECT HUNT
        </h1>

        <p style="text-align:center; color:#cbd5e1; font-size:18px;">
        Find a specific object or pattern inside an image
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # -------------------------------------------------
    # Upload section
    # -------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖼️ Main Image")
        main_file = st.file_uploader(
            "Upload the image to search",
            type=["jpg", "jpeg", "png"],
            key="main_image"
        )

    with col2:
        st.markdown("### 🎯 Target Template")
        template_file = st.file_uploader(
            "Upload the object/template to find",
            type=["jpg", "jpeg", "png"],
            key="template_image"
        )

    st.divider()

    # -------------------------------------------------
    # Show uploaded images
    # -------------------------------------------------

    if main_file and template_file:

        main_image = Image.open(main_file).convert("RGB")
        template_image = Image.open(template_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔎 Search Image")
            st.image(main_image, use_container_width=True)

        with col2:
            st.markdown("#### 🎯 Target Object")
            st.image(template_image, use_container_width=True)

        st.divider()

        # -------------------------------------------------
        # Scan button
        # -------------------------------------------------

        if st.button(
            "🚀 SCAN FOR MATCH",
            use_container_width=True,
            type="primary"
        ):

            # Convert PIL → OpenCV
            main_array = np.array(main_image)
            template_array = np.array(template_image)

            main_gray = cv2.cvtColor(
                main_array,
                cv2.COLOR_RGB2GRAY
            )

            template_gray = cv2.cvtColor(
                template_array,
                cv2.COLOR_RGB2GRAY
            )

            # Check template size
            main_height, main_width = main_gray.shape
            template_height, template_width = template_gray.shape

            if (
                template_height > main_height
                or template_width > main_width
            ):
                st.error(
                    "❌ Template image is larger than the main image."
                )
                return

            # ---------------------------------------------
            # Template Matching
            # ---------------------------------------------

            result = cv2.matchTemplate(
                main_gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED
            )

            min_value, max_value, min_location, max_location = cv2.minMaxLoc(
                result
            )

            similarity = max_value * 100

            # Best match location
            x, y = max_location

            # Draw rectangle
            output_image = main_array.copy()

            cv2.rectangle(
                output_image,
                (x, y),
                (
                    x + template_width,
                    y + template_height
                ),
                (0, 255, 0),
                4
            )

            # Add label
            cv2.putText(
                output_image,
                "BEST MATCH",
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ---------------------------------------------
            # Result
            # ---------------------------------------------

            st.markdown("## 📊 ANALYSIS RESULT")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "🎯 Similarity",
                    f"{similarity:.2f}%"
                )

            with metric2:
                st.metric(
                    "📍 X Position",
                    x
                )

            with metric3:
                st.metric(
                    "📍 Y Position",
                    y
                )

            st.divider()

            # ---------------------------------------------
            # Match status
            # ---------------------------------------------

            if similarity >= 70:

                st.success(
                    f"🟢 MATCH FOUND — Similarity {similarity:.2f}%"
                )

            else:

                st.warning(
                    f"🟡 LOW MATCH — Similarity {similarity:.2f}%"
                )

            # ---------------------------------------------
            # Result image
            # ---------------------------------------------

            st.markdown("### 🛰️ Object Hunt Result")

            st.image(
                output_image,
                caption="Best matching region highlighted",
                use_container_width=True
            )

            # ---------------------------------------------
            # Technical information
            # ---------------------------------------------

            with st.expander("🔬 View Technical Details"):

                st.write(
                    "Algorithm: OpenCV Template Matching"
                )

                st.write(
                    "Matching Method: TM_CCOEFF_NORMED"
                )

                st.write(
                    f"Template Size: "
                    f"{template_width} × {template_height} pixels"
                )

                st.write(
                    f"Detected Location: "
                    f"({x}, {y})"
                )

                st.write(
                    f"Similarity Score: "
                    f"{similarity:.2f}%"
                )

    else:

        st.info(
            "👆 Upload both the main image and target template "
            "to start the Object Hunt."
        )


def viola_jones_page():

    # ==================================================
    # BRIGHT TEXT + FUTURISTIC STYLING
    # ==================================================

    st.markdown("""
    <style>

        /* Main headings */
        h1 {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        h2, h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        /* Normal text */
        p {
            color: #d9e2ff !important;
        }

        /* Caption */
        .stCaption {
            color: #00f5ff !important;
        }

        /* File uploader label */
        label {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Markdown text */
        .stMarkdown {
            color: #ffffff;
        }

        /* Metric value */
        [data-testid="stMetricValue"] {
            color: #00f5ff !important;
            font-weight: 800 !important;
        }

        /* Metric label */
        [data-testid="stMetricLabel"] {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Alert text */
        .stAlert {
            color: #ffffff !important;
        }

        /* Button */
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            padding: 12px;
            font-weight: 700;
        }

    </style>
    """, unsafe_allow_html=True)


    # ==================================================
    # HEADER
    # ==================================================

    st.title("👤 FACE SCANNER")

    st.subheader(
        "Viola–Jones Face Detection Engine"
    )

    st.caption(
        "AI • REAL-TIME VISION • FACE DETECTION"
    )

    st.divider()


    # ==================================================
    # IMAGE UPLOAD
    # ==================================================

    uploaded_file = st.file_uploader(
        "📸 Upload a Face Image",
        type=["jpg", "jpeg", "png"]
    )


    # ==================================================
    # IMAGE AVAILABLE
    # ==================================================

    if uploaded_file is not None:

        # Read image
        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_np = np.array(image)


        # ==================================================
        # INPUT IMAGE
        # ==================================================

        st.subheader("🖼️ Input Image")

        st.image(
            image,
            caption="Uploaded Face Image",
            use_container_width=True
        )

        st.write("")


        # ==================================================
        # SCAN BUTTON
        # ==================================================

        scan_button = st.button(
            "🚀 START FACE SCAN",
            use_container_width=True
        )


        if scan_button:

            # ==================================================
            # PROCESSING
            # ==================================================

            with st.spinner(
                "🔎 Scanning image for faces..."
            ):

                # RGB → BGR
                frame = cv2.cvtColor(
                    image_np,
                    cv2.COLOR_RGB2BGR
                )


                # BGR → Gray
                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )


                # ==================================================
                # HAAR CASCADE
                # ==================================================

                cascade_path = (
                    cv2.data.haarcascades
                    + "haarcascade_frontalface_default.xml"
                )


                face_cascade = cv2.CascadeClassifier(
                    cascade_path
                )


                # Check classifier
                if face_cascade.empty():

                    st.error(
                        "❌ Haar Cascade could not be loaded."
                    )

                    return


                # ==================================================
                # FACE DETECTION
                # ==================================================

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50)
                )


                # Copy image
                result = frame.copy()


                # ==================================================
                # DRAW FACE BOXES
                # ==================================================

                for i, (x, y, w, h) in enumerate(faces):

                    # Green rectangle
                    cv2.rectangle(
                        result,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        3
                    )


                    # Face label
                    cv2.putText(
                        result,
                        f"FACE {i + 1}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )


                # BGR → RGB
                result_rgb = cv2.cvtColor(
                    result,
                    cv2.COLOR_BGR2RGB
                )


            # ==================================================
            # SCAN COMPLETE
            # ==================================================

            st.success(
                "✅ Face scanning completed successfully!"
            )

            st.divider()


            # ==================================================
            # RESULTS
            # ==================================================

            st.subheader(
                "📊 Scan Results"
            )


            # ==================================================
            # METRICS
            # ==================================================

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "👤 FACES DETECTED",
                    len(faces)
                )


            with col2:

                st.metric(
                    "⚙️ SCALE FACTOR",
                    "1.1"
                )


            with col3:

                st.metric(
                    "🎯 MIN NEIGHBORS",
                    "5"
                )


            st.divider()


            # ==================================================
            # DETECTION OUTPUT
            # ==================================================

            st.subheader(
                "🧠 Detection Output"
            )


            st.image(
                result_rgb,
                caption="Viola–Jones Face Detection Result",
                use_container_width=True
            )


            # ==================================================
            # FACE LOCATIONS
            # ==================================================

            if len(faces) > 0:

                st.subheader(
                    "📍 Face Locations"
                )


                for i, (x, y, w, h) in enumerate(faces):

                    st.info(
                        f"👤 FACE {i + 1}  |  "
                        f"X: {x}  |  "
                        f"Y: {y}  |  "
                        f"Width: {w}  |  "
                        f"Height: {h}"
                    )


            else:

                st.warning(
                    "⚠️ No face detected. "
                    "Please try a clear front-facing image."
                )


            # ==================================================
            # TECHNICAL DETAILS
            # ==================================================

            with st.expander(
                "🔬 Technical Details"
            ):

                st.write(
                    "Algorithm: Viola–Jones"
                )

                st.write(
                    "Classifier: Haar Cascade"
                )

                st.write(
                    "Detection Method: detectMultiScale"
                )

                st.write(
                    "Scale Factor: 1.1"
                )

                st.write(
                    "Minimum Neighbors: 5"
                )

                st.write(
                    "Minimum Face Size: 50 × 50 pixels"
                )


    # ==================================================
    # NO IMAGE
    # ==================================================

    else:

        st.info(
            "📸 Upload a face image to activate "
            "the Face Scanner."
        )


def deepface_analysis_page():

    st.title("🧠 AI FACE INSIGHT")
    st.subheader("DeepFace Facial Analysis Engine")
    st.caption("AI • DEEP LEARNING • FACE ANALYSIS")

    st.divider()

    uploaded_file = st.file_uploader(
        "📸 Upload a Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("🖼️ Input Image")

        st.image(
            image,
            caption="Uploaded Face Image",
            width="stretch"
        )

        st.write("")

        analyze_button = st.button(
            "🚀 ANALYZE FACE",
            width="stretch"
        )

        if analyze_button:

            with st.spinner(
                "🧠 DeepFace is analyzing the face..."
            ):

                temp_path = None

                try:

                    # Create temporary image file
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".jpg"
                    ) as temp_file:

                        image.save(
                            temp_file,
                            format="JPEG"
                        )

                        temp_path = temp_file.name

                    # DeepFace emotion analysis
                    result = DeepFace.analyze(
                        img_path=temp_path,
                        actions=["emotion"],
                        enforce_detection=False
                    )

                    # Handle different DeepFace return formats
                    if isinstance(result, list):
                        result = result[0]

                    dominant_emotion = result.get(
                        "dominant_emotion",
                        "Unknown"
                    )

                    emotion_scores = result.get(
                        "emotion",
                        {}
                    )

                except Exception as e:

                    st.error(
                        "❌ DeepFace analysis failed."
                    )

                    st.caption(str(e))
                    return

                finally:

                    if (
                        temp_path
                        and os.path.exists(temp_path)
                    ):
                        os.remove(temp_path)

            st.success(
                "✅ Face analysis completed successfully!"
            )

            st.divider()

            st.subheader("📊 Analysis Result")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "😊 DOMINANT EMOTION",
                    dominant_emotion.upper()
                )

            with col2:

                if emotion_scores:

                    confidence = max(
                        emotion_scores.values()
                    )

                    st.metric(
                        "🎯 CONFIDENCE",
                        f"{confidence:.1f}%"
                    )

                else:

                    st.metric(
                        "🎯 CONFIDENCE",
                        "N/A"
                    )

            st.divider()

            st.subheader("🧠 Emotion Analysis")

            if emotion_scores:

                for emotion, score in emotion_scores.items():

                    st.write(
                        f"**{emotion.capitalize()}** "
                        f"— {score:.1f}%"
                    )

                    st.progress(
                        min(int(score), 100)
                    )

            st.divider()

            with st.expander(
                "🔬 Technical Details"
            ):

                st.write(
                    "Model: DeepFace"
                )

                st.write(
                    "Analysis Type: Facial Emotion Analysis"
                )

                st.write(
                    "Method: Deep Learning"
                )

                st.write(
                    "Detection: Automatic Face Detection"
                )

    else:

        st.info(
            "📸 Upload a face image to activate "
            "the AI Face Insight engine."
        )


def facenet_analysis_page():

    st.title("📐 FACE SIMILARITY LAB")
    st.subheader("FaceNet Similarity Engine")
    st.caption("AI • FACE EMBEDDINGS • SIMILARITY ANALYSIS")

    st.divider()

    st.write(
        "Upload two face images to compare their "
        "facial feature similarity."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("👤 FACE 01")

        image1_file = st.file_uploader(
            "Upload First Face",
            type=["jpg", "jpeg", "png"],
            key="face1"
        )

        if image1_file is not None:

            image1 = Image.open(
                image1_file
            ).convert("RGB")

            st.image(
                image1,
                caption="Face 01",
                width="stretch"
            )

    with col2:

        st.subheader("👤 FACE 02")

        image2_file = st.file_uploader(
            "Upload Second Face",
            type=["jpg", "jpeg", "png"],
            key="face2"
        )

        if image2_file is not None:

            image2 = Image.open(
                image2_file
            ).convert("RGB")

            st.image(
                image2,
                caption="Face 02",
                width="stretch"
            )

    st.write("")

    compare_button = st.button(
        "🚀 COMPARE FACES",
        width="stretch"
    )

    if compare_button:

        if image1_file is None or image2_file is None:

            st.warning(
                "⚠️ Please upload both face images."
            )

            return

        with st.spinner(
            "🧠 FaceNet is generating facial embeddings..."
        ):

            try:

                # Convert images to NumPy arrays
                img1 = np.array(image1)
                img2 = np.array(image2)

                # Load FaceNet model
                embedder = FaceNet()

                # Generate embeddings
                embedding1 = embedder.embeddings(
                    [img1]
                )[0]

                embedding2 = embedder.embeddings(
                    [img2]
                )[0]

                # Calculate cosine distance
                distance = cosine(
                    embedding1,
                    embedding2
                )

                # Convert distance to similarity
                similarity = (
                    1 - distance
                ) * 100

                similarity = max(
                    0,
                    min(100, similarity)
                )

            except Exception as e:

                st.error(
                    "❌ FaceNet comparison failed."
                )

                st.caption(str(e))

                return

        st.success(
            "✅ Face comparison completed!"
        )

        st.divider()

        st.subheader(
            "📊 Similarity Result"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📐 SIMILARITY",
                f"{similarity:.1f}%"
            )

        with col2:

            st.metric(
                "📏 COSINE DISTANCE",
                f"{distance:.3f}"
            )

        with col3:

            st.metric(
                "🧠 EMBEDDING SIZE",
                str(len(embedding1))
            )

        st.divider()

        st.subheader(
            "🎯 Comparison Status"
        )

        st.progress(
            int(similarity)
        )

        if similarity >= 70:

            st.success(
                "🟢 HIGH FEATURE SIMILARITY"
            )

            st.write(
                "The two images show a high level "
                "of facial feature similarity."
            )

        elif similarity >= 50:

            st.warning(
                "🟡 MODERATE FEATURE SIMILARITY"
            )

            st.write(
                "The two images show moderate "
                "facial feature similarity."
            )

        else:

            st.info(
                "🔵 LOW FEATURE SIMILARITY"
            )

            st.write(
                "The two images show low "
                "facial feature similarity."
            )

        st.divider()

        with st.expander(
            "🔬 Technical Details"
        ):

            st.write(
                "Model: FaceNet"
            )

            st.write(
                "Feature Representation: Face Embeddings"
            )

            st.write(
                "Comparison Method: Cosine Distance"
            )

            st.write(
                "Output: Feature Similarity Score"
            )


import streamlit as st



# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="NEXUS VISION",
    page_icon="🧿",
    layout="wide"
)


# ==================================================
# SESSION STATE
# ==================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# ==================================================
# NEXUS VISION DESIGN
# ==================================================

st.markdown(
    """
    <style>

    /* MAIN BACKGROUND */
    .stApp {
        background:
        linear-gradient(
            135deg,
            #030712 0%,
            #081126 50%,
            #101a38 100%
        );
    }

    /* HEADINGS */
    h1 {
        color: #ffffff !important;
        font-weight: 900 !important;
    }

    h2 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* NORMAL TEXT */
    p {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* CAPTIONS */
    .stCaption {
        color: #00f5ff !important;
        font-weight: 600 !important;
    }

    /* MARKDOWN */
    .stMarkdown {
        color: #ffffff !important;
    }

    .stMarkdown p {
        color: #ffffff !important;
    }

    /* FILE UPLOADER */
    label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* METRICS */
    [data-testid="stMetricValue"] {
        color: #00f5ff !important;
        font-weight: 900 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ALERTS */
    .stAlert {
        color: #ffffff !important;
    }

    /* BUTTONS */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 12px 18px;
        font-weight: 800;
        color: #ffffff;
        background:
        linear-gradient(
            90deg,
            #111c3a,
            #172554
        );
        border: 1px solid #00f5ff;
        transition: 0.3s;
    }

    .stButton > button:hover {
        border: 1px solid #ffffff;
        transform: translateY(-2px);
        background:
        linear-gradient(
            90deg,
            #172554,
            #1e3a8a
        );
    }

    /* DIVIDERS */
    hr {
        border-color: #24345f !important;
    }

    /* INPUT TEXT */
    input {
        color: #ffffff !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HOME DASHBOARD
# ==================================================

if st.session_state.page == "home":

    st.title("🧿 NEXUS VISION")

    st.subheader(
        "Where Vision Meets Intelligence"
    )

    st.caption(
        "AI • COMPUTER VISION • FACE ANALYTICS"
    )

    st.success(
        "🟢 VISION ENGINE ONLINE • SYSTEM READY"
    )

    st.divider()


    # ==================================================
    # SYSTEM METRICS
    # ==================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "VISION MODULES",
            "04"
        )

    with col2:
        st.metric(
            "AI ENGINE",
            "ACTIVE"
        )

    with col3:
        st.metric(
            "CV TECHNOLOGY",
            "READY"
        )

    with col4:
        st.metric(
            "SYSTEM STATUS",
            "ONLINE"
        )


    st.divider()


    # ==================================================
    # MODULE SECTION
    # ==================================================

    st.subheader(
        "🚀 Vision Intelligence Modules"
    )

    st.write(
        "Choose a module to explore the computer "
        "vision technology."
    )

    st.write("")


    # ==================================================
    # ROW 1
    # ==================================================

    col1, col2 = st.columns(2)


    # TEMPLATE MATCHING
    with col1:

        st.markdown(
            "### 🔍 OBJECT HUNT"
        )

        st.write(
            "Template Matching"
        )

        st.caption(
            "Find a specific object or pattern "
            "inside an image."
        )

        if st.button(
            "START OBJECT HUNT  →",
            key="template_button"
        ):

            st.session_state.page = "template"
            st.rerun()


    # VIOLA JONES
    with col2:

        st.markdown(
            "### 👤 FACE SCANNER"
        )

        st.write(
            "Viola–Jones Algorithm"
        )

        st.caption(
            "Detect faces using Haar Cascade "
            "face detection."
        )

        if st.button(
            "START FACE SCANNER  →",
            key="viola_button"
        ):

            st.session_state.page = "viola"
            st.rerun()


    st.write("")


    # ==================================================
    # ROW 2
    # ==================================================

    col1, col2 = st.columns(2)


    # DEEPFACE
    with col1:

        st.markdown(
            "### 🧠 AI FACE INSIGHT"
        )

        st.write(
            "DeepFace"
        )

        st.caption(
            "Analyze facial emotion using "
            "deep learning."
        )

        if st.button(
            "START AI ANALYSIS  →",
            key="deepface_button"
        ):

            st.session_state.page = "deepface"
            st.rerun()


    # FACENET
    with col2:

        st.markdown(
            "### 📐 FACE SIMILARITY LAB"
        )

        st.write(
            "FaceNet"
        )

        st.caption(
            "Compare facial features using "
            "deep learning embeddings."
        )

        if st.button(
            "START FACE COMPARISON  →",
            key="facenet_button"
        ):

            st.session_state.page = "facenet"
            st.rerun()


    st.divider()


    # ==================================================
    # FOOTER
    # ==================================================

    st.caption(
        "NEXUS VISION  •  COMPUTER VISION LAB"
    )


# ==================================================
# TEMPLATE MATCHING PAGE
# ==================================================

elif st.session_state.page == "template":

    if st.button(
        "← BACK TO NEXUS VISION",
        key="back_template"
    ):

        st.session_state.page = "home"
        st.rerun()

    st.divider()

    template_matching_page()


# ==================================================
# VIOLA–JONES PAGE
# ==================================================

elif st.session_state.page == "viola":

    if st.button(
        "← BACK TO NEXUS VISION",
        key="back_viola"
    ):

        st.session_state.page = "home"
        st.rerun()

    st.divider()

    viola_jones_page()


# ==================================================
# DEEPFACE PAGE
# ==================================================

elif st.session_state.page == "deepface":

    if st.button(
        "← BACK TO NEXUS VISION",
        key="back_deepface"
    ):

        st.session_state.page = "home"
        st.rerun()

    st.divider()

    deepface_analysis_page()


# ==================================================
# FACENET PAGE
# ==================================================

elif st.session_state.page == "facenet":

    if st.button(
        "← BACK TO NEXUS VISION",
        key="back_facenet"
    ):

        st.session_state.page = "home"
        st.rerun()

    st.divider()

    facenet_analysis_page()

