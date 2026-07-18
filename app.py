import os
import tempfile
import warnings

import cv2
import dlib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from imageio.v3 import imread
from skimage.transform import rescale, estimate_transform, warp

from Networks.predictor import MobilenetPosPredictor
from Utils.write import write_obj_with_colors

warnings.filterwarnings('ignore', category=DeprecationWarning)

# --- Page config ---
st.set_page_config(
    page_title="3D Face Reconstruction",
    page_icon="🧑",
    layout="wide"
)

st.title("🧑 3D Face Reconstruction")
st.markdown(
    "Upload a **frontal** and **side** face image, or just a **single** image to reconstruct a 3D face mesh."
)


# --- Cached model loading ---
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    model_path = 'Data/net-data/trained_fg_then_real.h5'
    face_detector_path = 'Data/net-data/mmod_human_face_detector.dat'
    shape_predictor_path = 'Data/net-data/shape_predictor_68_face_landmarks.dat'

    # Auto-download model files if not present (for cloud deployment)
    os.makedirs('Data/net-data', exist_ok=True)
    if not os.path.exists(model_path) or not os.path.exists(shape_predictor_path):
        try:
            import gdown
            gdown.download_folder(
                'https://drive.google.com/drive/folders/13Y8zCnvccDq7bwSuwGoQWawFxlD-tCMX',
                output='Data/net-data/', quiet=False
            )
        except Exception as e:
            st.error(f"Failed to download model files: {e}. Please add them manually to Data/net-data/")
            st.stop()

    face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
    shape_predictor = dlib.shape_predictor(shape_predictor_path)

    pos_predictor = MobilenetPosPredictor(256, 256)
    pos_predictor.restore(model_path)

    triangles = np.loadtxt('Data/uv-data/triangles.txt').astype(np.int32)
    face_ind = np.loadtxt('Data/uv-data/face_ind.txt').astype(np.int32)

    return face_detector, shape_predictor, pos_predictor, triangles, face_ind


# --- Helper functions (from demo.py) ---
def mask_pos(pos):
    mask_path = 'Data/uv-data/facegen_face_mask.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    index_mask = mask[:, :] < 0.5
    masked_pos = pos.copy()
    masked_pos[index_mask] = [0, 0, 0]
    return masked_pos


def get_cropping_transformation(image, face_detector, shape_predictor):
    detected_faces = face_detector(image, 1)
    if len(detected_faces) == 0:
        return None

    d = detected_faces[0].rect
    left = d.left()
    right = d.right()
    top = d.top()
    bottom = d.bottom()
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
    size = int(old_size * 1.58)

    shape = shape_predictor(image, d)
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2]
    ])
    DST_PTS = np.array([[0, 0], [0, 255], [255, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    return coords, tform


def uncrop_pos(cropped_pos, cropping_tform):
    cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
    z = cropped_vertices[2, :].copy() / cropping_tform.params[0, 0]
    cropped_vertices[2, :] = 1
    vertices = np.dot(np.linalg.inv(cropping_tform.params), cropped_vertices)
    vertices = np.vstack((vertices[:2, :], z))
    pos = np.reshape(vertices.T, [256, 256, 3])
    return pos


def get_cropped_image(img, cropping_tform):
    float_img = img / 256.0 / 1.1
    if cropping_tform is None:
        return float_img
    else:
        return warp(float_img, cropping_tform.inverse, output_shape=(256, 256))


def plot_vertices_on_image_from_pos(pos, l68, front_img):
    h, w, c = pos.shape
    plotted_front_img = front_img.copy().astype(np.uint8)
    h_i, w_i, c_i = plotted_front_img.shape
    max_h = np.max(pos[:, :, 1])
    max_w = np.max(pos[:, :, 0])
    max_z = int(np.max(pos[:, :, 2]))

    if (max_w - w_i) > 0:
        enlarged = np.zeros((h_i, int(max_w), 3), dtype=np.uint8)
        enlarged[:, 0:w_i, :] = plotted_front_img
        plotted_front_img = enlarged
        h_i, w_i, c_i = plotted_front_img.shape

    if (max_h - h_i) > 0:
        enlarged = np.zeros((int(max_h), w_i, 3), dtype=np.uint8)
        enlarged[0:h_i, :, :] = plotted_front_img
        plotted_front_img = enlarged
        h_i, w_i, c_i = plotted_front_img.shape

    for h_u in range(h):
        for w_u in range(w):
            index = np.around(pos[h_u][w_u], decimals=1).astype(int)
            if 0 <= index[1] - 2 < h_i and 0 <= index[0] - 2 < w_i:
                plotted_front_img[index[1] - 2][index[0] - 2] = [0, 255 - (max_z - index[2]), index[2]]

    for (x, y) in l68:
        if 1 <= y < h_i - 1 and 1 <= x < w_i - 1:
            plotted_front_img[y][x] = [255, 0, 0]
            plotted_front_img[y + 1][x + 1] = [255, 0, 0]
            plotted_front_img[y + 1][x - 1] = [255, 0, 0]
            plotted_front_img[y - 1][x + 1] = [255, 0, 0]
            plotted_front_img[y - 1][x - 1] = [255, 0, 0]
    return plotted_front_img


def process_images(front_img, side_img, face_detector, shape_predictor, pos_predictor, triangles, face_ind):
    """Run the full reconstruction pipeline. Returns (obj_content, projected_img, cropped_front, cropped_side, vertices_3d, colors_rgb)."""

    # Resize if too large
    if front_img.shape != (256, 256, 3):
        max_size = max(front_img.shape[0], front_img.shape[1])
        if max_size > 1000:
            front_img = rescale(front_img, 1000. / max_size, channel_axis=2)
            front_img = (front_img * 255).astype(np.uint8)
        front_img = np.around(front_img, decimals=1).astype(np.uint8)

    if side_img.shape != (256, 256, 3):
        max_size = max(side_img.shape[0], side_img.shape[1])
        if max_size > 1000:
            side_img = rescale(side_img, 1000. / max_size, channel_axis=2)
            side_img = (side_img * 255).astype(np.uint8)
        side_img = np.around(side_img, decimals=1).astype(np.uint8)

    # Detect faces and get cropping transforms
    result_front = get_cropping_transformation(front_img, face_detector, shape_predictor)
    if result_front is None:
        raise ValueError("No face detected in the front image. Please try a different image.")
    l68_front, cropping_tform_front = result_front

    result_side = get_cropping_transformation(side_img, face_detector, shape_predictor)
    if result_side is None:
        raise ValueError("No face detected in the side image. Please try a different image.")
    l68_side, cropping_tform_side = result_side

    # Crop and predict
    cropped_image_front = get_cropped_image(front_img, cropping_tform_front)
    cropped_image_side = get_cropped_image(side_img, cropping_tform_side)

    img_concat = np.concatenate((cropped_image_front, cropped_image_side), axis=2)
    cropped_pos = pos_predictor.predict(img_concat)

    # Uncrop position map
    pos = uncrop_pos(cropped_pos, cropping_tform_front)

    # Extract vertices
    all_vertices = np.reshape(pos, [256 ** 2, -1])
    vertices = all_vertices[face_ind, :]

    save_vertices = vertices.copy()
    save_vertices[:, 1] = 256 - 1 - save_vertices[:, 1]

    # Get colors from image
    [h, w, _] = front_img.shape
    vertices[:, 0] = np.minimum(np.maximum(vertices[:, 0], 0), w - 1)
    vertices[:, 1] = np.minimum(np.maximum(vertices[:, 1], 0), h - 1)
    ind = np.round(vertices).astype(np.int32)
    colors = front_img[ind[:, 1], ind[:, 0], :]

    # Write OBJ to temp file
    tmp_obj_path = tempfile.mktemp(suffix='.obj')
    write_obj_with_colors(tmp_obj_path, save_vertices, triangles, colors)
    with open(tmp_obj_path, 'r') as f:
        obj_content = f.read()
    os.unlink(tmp_obj_path)

    # Generate projected visualization
    masked_pos = mask_pos(pos)
    projected_img = plot_vertices_on_image_from_pos(masked_pos, l68_front, front_img)

    # Convert cropped images for display
    cropped_front_display = (np.clip(cropped_image_front, 0, 1) * 255).astype(np.uint8)
    cropped_side_display = (np.clip(cropped_image_side, 0, 1) * 255).astype(np.uint8)

    return obj_content, projected_img, cropped_front_display, cropped_side_display, save_vertices, colors


# --- Load models ---
face_detector, shape_predictor, pos_predictor, triangles, face_ind = load_models()


def create_3d_point_cloud(vertices, colors):
    """Create an interactive 3D point cloud visualization using Plotly with interpolated mesh surface."""
    # Convert colors to normalized [0,1] for plotly
    colors_norm = colors.astype(np.float64) / 255.0

    # Create interpolated mesh surface using Mesh3d with intensity-based coloring
    # Use Delaunay triangulation via plotly's alphahull or provide triangles
    fig = go.Figure()

    # Add mesh3d with vertex colors via facecolor interpolation
    # We use i,j,k from the triangles array for proper mesh rendering
    from scipy.spatial import Delaunay

    # Project to 2D (x,y) for triangulation since it's a face surface
    points_2d = vertices[:, :2]
    try:
        tri = Delaunay(points_2d)
        simplices = tri.simplices

        # Compute face colors by averaging vertex colors for each triangle
        face_colors = []
        for s in simplices:
            r = int(np.mean(colors[s, 0]))
            g = int(np.mean(colors[s, 1]))
            b = int(np.mean(colors[s, 2]))
            face_colors.append(f'rgb({r},{g},{b})')

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=simplices[:, 0],
            j=simplices[:, 1],
            k=simplices[:, 2],
            facecolor=face_colors,
            flatshading=False,
            lighting=dict(ambient=0.7, diffuse=0.5, specular=0.2, roughness=0.5),
            lightposition=dict(x=100, y=100, z=200),
            opacity=1.0,
        ))
    except Exception:
        # Fallback to point cloud if triangulation fails
        color_strings = [f'rgb({r},{g},{b})' for r, g, b in colors]
        fig.add_trace(go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode='markers',
            marker=dict(size=1.5, color=color_strings, opacity=0.9),
        ))

    fig.update_layout(
        title="3D Face Reconstruction (interactive - drag to rotate)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=0, y=0, z=-2)),
        ),
        width=800,
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig

# --- Sidebar: choose input mode ---
st.sidebar.header("Input")
input_mode = st.sidebar.radio("Choose input source:", ["Upload images (pair)", "Upload single image", "Use test images"])

front_img = None
side_img = None
single_mode = False

if input_mode == "Upload images (pair)":
    front_file = st.sidebar.file_uploader("Front face image", type=["jpg", "jpeg", "png"], key="front_upload")
    side_file = st.sidebar.file_uploader("Side face image", type=["jpg", "jpeg", "png"], key="side_upload")

    if front_file and side_file:
        front_img = imread(front_file)[:, :, :3]
        side_img = imread(side_file)[:, :, :3]

elif input_mode == "Upload single image":
    single_file = st.sidebar.file_uploader("Face image", type=["jpg", "jpeg", "png"], key="single_upload")
    st.sidebar.caption("The same image will be used as both front and side input.")

    if single_file:
        front_img = imread(single_file)[:, :, :3]
        side_img = front_img.copy()
        single_mode = True

else:
    # List available test folders
    test_folders = [f for f in os.listdir('test_images') if f != 'results' and os.path.isdir(os.path.join('test_images', f))]
    test_folders.sort()
    selected = st.sidebar.selectbox("Select test image pair:", test_folders)

    if selected:
        front_path = os.path.join('test_images', selected, 'front.jpg')
        side_path = os.path.join('test_images', selected, 'side.jpg')
        if os.path.exists(front_path) and os.path.exists(side_path):
            front_img = imread(front_path)[:, :, :3]
            side_img = imread(side_path)[:, :, :3]

# --- Show inputs and run reconstruction ---
if front_img is not None and side_img is not None:
    if single_mode:
        col1, col3 = st.columns(2)
        with col1:
            st.image(front_img, caption="Input Image", width='stretch')
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(front_img, caption="Front", width='stretch')
        with col2:
            st.image(side_img, caption="Side", width='stretch')

    if st.button("🔄 Reconstruct 3D Face", type="primary", width='stretch'):
        with st.spinner("Processing... (this may take a moment on CPU)"):
            try:
                obj_content, projected_img, cropped_front, cropped_side, vertices_3d, colors_rgb = process_images(
                    front_img, side_img, face_detector, shape_predictor,
                    pos_predictor, triangles, face_ind
                )

                st.session_state['reconstruction_done'] = True
                st.session_state['vertices_3d'] = vertices_3d
                st.session_state['colors_rgb'] = colors_rgb

            except ValueError as e:
                st.error(str(e))
                st.session_state['reconstruction_done'] = False
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state['reconstruction_done'] = False

    # Show results if available
    if st.session_state.get('reconstruction_done', False):
        with col3:
            fig = create_3d_point_cloud(st.session_state['vertices_3d'], st.session_state['colors_rgb'])
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Upload front and side face images, or select a test image pair from the sidebar.")
