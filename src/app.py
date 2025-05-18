import os
import json
import time
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

# Import utility modules
from utils.audio_analyzer import analyze_audio
from utils.storyboard import generate_storyboard
from utils.media_generator import (
    generate_images,
    generate_videos_from_images,
    create_music_video,
    preview_scene
)
from utils.visualization import (
    create_audio_visualizations,
    create_lyrics_visualization,
    create_sections_visualization,
    create_audio_properties_markdown,
    create_storyboard_visualization,
    create_color_palette_visualization,
    create_timeline_visualization
)

# Load environment variables
load_dotenv()

# Check for API keys
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in .env file")
if not os.getenv("HUGGINGFACE_TOKEN"):
    print("Warning: HUGGINGFACE_TOKEN not found in .env file")

# Define paths
ASSETS_DIR = Path("assets")
SAMPLES_DIR = ASSETS_DIR / "samples"
GENERATED_DIR = ASSETS_DIR / "generated"
STEMS_DIR = GENERATED_DIR / "stems"
IMAGES_DIR = GENERATED_DIR / "images"
VIDEOS_DIR = GENERATED_DIR / "videos"

# Ensure directories exist
for dir_path in [SAMPLES_DIR, STEMS_DIR, IMAGES_DIR, VIDEOS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Define theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_200",
    button_secondary_text_color="*neutral_800",
    block_title_text_color="*primary_500",
    block_title_background_fill="*neutral_50",
    input_background_fill="white",
    container_radius="*radius_sm",
    tab_selected_background_fill="*primary_100",
    tab_selected_text_color="*primary_600"
)

# Global state to store analysis results and storyboard
class AppState:
    def __init__(self):
        self.audio_path = None
        self.analysis_results = None
        self.storyboard = None
        self.generated_images = []
        self.generated_videos = []
        self.final_video = None

# Initialize app state
app_state = AppState()

def process_audio(audio_file, progress=gr.Progress()):
    """Process uploaded audio file and extract information"""
    if audio_file is None:
        return (
            "Please upload an audio file.",
            None, None, None, None, None, None, None
        )
    
    try:
        # Store audio path
        app_state.audio_path = audio_file
        
        # Update progress
        progress(0.1, "Starting audio analysis...")
        
        # Analyze audio
        progress(0.2, "Analyzing audio features...")
        analysis_results = analyze_audio(audio_file)
        
        # Store analysis results
        app_state.analysis_results = analysis_results
        
        # Update progress
        progress(0.8, "Creating visualizations...")
        
        # Create visualizations
        visualizations = create_audio_visualizations(analysis_results)
        lyrics_df = create_lyrics_visualization(analysis_results.get("lyrics_segments", []))
        sections_df = create_sections_visualization(analysis_results.get("sections", []))
        properties_md = create_audio_properties_markdown(analysis_results)
        
        # Create stems display
        stems = analysis_results.get("stems", {})
        stems_html = "<div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>"
        stems_html += "<h3>Separated Audio Stems</h3>"
        
        if stems:
            stems_html += "<ul>"
            for stem_name, stem_path in stems.items():
                stems_html += f"<li><strong>{stem_name}</strong>: <audio src='file/{stem_path}' controls></audio></li>"
            stems_html += "</ul>"
        else:
            stems_html += "<p>No stems available. Please try again.</p>"
        
        stems_html += "</div>"
        
        # Update progress
        progress(1.0, "Analysis complete!")
        
        # Format results for display
        markdown_output = f"""
        ## Audio Analysis Results
        
        ### Lyrics
        ```
        {analysis_results.get('lyrics', 'No lyrics detected')}
        ```
        """
        
        return (
            markdown_output,
            visualizations.get("waveform"),
            visualizations.get("spectrogram"),
            visualizations.get("chromagram"),
            lyrics_df,
            sections_df,
            properties_md,
            stems_html
        )
    
    except Exception as e:
        error_message = f"Error analyzing audio: {str(e)}"
        print(error_message)
        return (
            error_message,
            None, None, None, None, None, None, None
        )

def create_storyboard_from_analysis(theme, style_preferences, progress=gr.Progress()):
    """Generate storyboard based on audio analysis and theme"""
    if app_state.audio_path is None or app_state.analysis_results is None:
        return "Please analyze an audio file first.", None, None
    
    try:
        # Update progress
        progress(0.1, "Starting storyboard generation...")
        
        # Parse style preferences
        style_prefs = {}
        if style_preferences:
            lines = style_preferences.strip().split("\n")
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    style_prefs[key.strip()] = value.strip()
        
        # Generate storyboard
        progress(0.3, "Generating storyboard with AI...")
        storyboard = generate_storyboard(
            app_state.analysis_results,
            theme=theme if theme else "Default theme",
            style_preferences=style_prefs
        )
        
        # Store storyboard
        app_state.storyboard = storyboard
        
        # Update progress
        progress(0.7, "Creating visualizations...")
        
        # Create storyboard visualization
        storyboard_df = create_storyboard_visualization(storyboard)
        
        # Create timeline visualization
        timeline = create_timeline_visualization(
            storyboard,
            app_state.analysis_results.get("duration", 180)
        )
        
        # Update progress
        progress(1.0, "Storyboard generation complete!")
        
        return "Storyboard generated successfully!", storyboard_df, timeline
    
    except Exception as e:
        error_message = f"Error generating storyboard: {str(e)}"
        print(error_message)
        return error_message, None, None

def preview_storyboard_scene(scene_index, progress=gr.Progress()):
    """Preview a single scene from the storyboard"""
    if app_state.storyboard is None or not app_state.storyboard:
        return "Please generate a storyboard first.", None, None, None
    
    try:
        # Get scene data
        scene_index = int(scene_index)
        if scene_index < 0 or scene_index >= len(app_state.storyboard):
            return f"Invalid scene index: {scene_index}", None, None, None
        
        scene = app_state.storyboard[scene_index]
        
        # Update progress
        progress(0.1, f"Previewing scene {scene_index + 1}...")
        
        # Extract scene data
        prompt = scene.get("visual_prompt", "")
        time_str = scene.get("time", "0s")
        description = scene.get("description", "")
        camera_movement = scene.get("camera_movement", "")
        color_palette = scene.get("color_palette", [])
        
        # Convert time to seconds
        if isinstance(time_str, str) and time_str.endswith("s"):
            time_sec = float(time_str[:-1])
        else:
            time_sec = float(time_str)
        
        # Create audio segment for preview (if available)
        audio_segment_path = None
        if app_state.audio_path:
            try:
                # Extract a short segment around the scene time
                import moviepy.editor as mp
                
                # Load audio
                audio = mp.AudioFileClip(app_state.audio_path)
                
                # Determine segment start and end
                segment_start = max(0, time_sec - 1)
                segment_end = min(audio.duration, time_sec + 4)
                
                # Extract segment
                segment = audio.subclip(segment_start, segment_end)
                
                # Save segment
                segment_path = STEMS_DIR / f"preview_segment_{scene_index}.mp3"
                segment.write_audiofile(str(segment_path))
                
                audio_segment_path = str(segment_path)
            except Exception as e:
                print(f"Error creating audio segment: {str(e)}")
        
        # Update progress
        progress(0.3, "Generating scene preview...")
        
        # Generate preview
        preview_result = preview_scene(
            prompt,
            audio_segment_path=audio_segment_path,
            style_params={
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 768
            }
        )
        
        # Create color palette visualization
        progress(0.8, "Creating color palette visualization...")
        palette_vis = create_color_palette_visualization(color_palette)
        
        # Format scene details
        scene_details = f"""
        ## Scene {scene_index + 1} Details
        
        **Time:** {time_str}
        
        **Description:** {description}
        
        **Visual Prompt:** {prompt}
        
        **Camera Movement:** {camera_movement}
        """
        
        # Update progress
        progress(1.0, "Preview generation complete!")
        
        return (
            scene_details,
            preview_result.get("image"),
            preview_result.get("video"),
            palette_vis
        )
    
    except Exception as e:
        error_message = f"Error previewing scene: {str(e)}"
        print(error_message)
        return error_message, None, None, None

def generate_music_video(transition_type, progress=gr.Progress()):
    """Generate the complete music video"""
    if app_state.audio_path is None:
        return "Please upload an audio file first.", None
    
    if app_state.storyboard is None or not app_state.storyboard:
        return "Please generate a storyboard first.", None
    
    try:
        # Update progress
        progress(0.05, "Starting music video generation...")
        
        # Generate images for each scene
        progress(0.1, "Generating images for each scene...")
        prompts = [scene.get("visual_prompt", "") for scene in app_state.storyboard]
        
        # Generate images in batches to avoid memory issues
        all_image_paths = []
        batch_size = 3  # Process 3 prompts at a time
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            progress(0.1 + 0.3 * (i / len(prompts)), f"Generating images {i+1}-{min(i+batch_size, len(prompts))} of {len(prompts)}...")
            
            batch_image_paths = generate_images(batch_prompts)
            all_image_paths.extend(batch_image_paths)
        
        # Store generated images
        app_state.generated_images = all_image_paths
        
        # Generate videos from images
        progress(0.4, "Generating videos from images...")
        all_video_paths = []
        
        for i, image_path in enumerate(all_image_paths):
            progress(0.4 + 0.3 * (i / len(all_image_paths)), f"Generating video {i+1} of {len(all_image_paths)}...")
            
            video_paths = generate_videos_from_images(
                [image_path],
                duration_per_video=5.0,
                fps=24
            )
            
            if video_paths:
                all_video_paths.extend(video_paths)
        
        # Store generated videos
        app_state.generated_videos = all_video_paths
        
        # Create transitions list based on selected type
        transitions = None
        if transition_type == "fade":
            transitions = ["fade"] * len(all_video_paths)
        elif transition_type == "slide":
            transitions = ["slide"] * len(all_video_paths)
        elif transition_type == "mixed":
            transitions = ["fade", "slide"] * (len(all_video_paths) // 2 + 1)
        
        # Create final music video
        progress(0.7, "Creating final music video...")
        final_video_path = create_music_video(
            all_video_paths,
            app_state.audio_path,
            transitions=transitions
        )
        
        # Store final video path
        app_state.final_video = final_video_path
        
        # Update progress
        progress(1.0, "Music video generation complete!")
        
        return "Music video generated successfully!", final_video_path
    
    except Exception as e:
        error_message = f"Error generating music video: {str(e)}"
        print(error_message)
        return error_message, None

# Create Gradio interface
with gr.Blocks(title="VisualSync Professional", theme=theme) as app:
    gr.Markdown(
        """
        # VisualSync Professional
        
        A professional-grade music video generation tool using AI for audio analysis, storyboarding, and visual creation.
        
        ## Workflow
        1. **Analysis Tab**: Upload your audio file for analysis and stem separation
        2. **Storyboard Tab**: Generate and customize your music video storyboard
        3. **Generation Tab**: Preview individual scenes and generate the complete music video
        """
    )
    
    # Store references to components that need to be accessed across tabs
    audio_input_ref = gr.Audio(label="Upload Audio", type="filepath", visible=False)
    
    with gr.Tabs() as tabs:
        # Tab 1: Audio Analysis
        with gr.TabItem("Analysis", id="analysis_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="Upload Audio",
                        type="filepath",
                        elem_id="audio_input"
                    )
                    analyze_button = gr.Button("Analyze Audio", variant="primary")
                    
                    with gr.Accordion("Audio Properties", open=False):
                        audio_properties = gr.Markdown()
                
                with gr.Column(scale=2):
                    analysis_output = gr.Markdown(label="Analysis Results")
            
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Waveform", open=True):
                        waveform_plot = gr.Plot(label="Waveform Visualization")
                
                with gr.Column():
                    with gr.Accordion("Spectrogram", open=False):
                        spectrogram_plot = gr.Plot(label="Spectrogram Visualization")
            
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Chromagram", open=False):
                        chromagram_plot = gr.Plot(label="Chromagram Visualization")
                
                with gr.Column():
                    with gr.Accordion("Audio Sections", open=False):
                        sections_df = gr.Dataframe(label="Audio Sections")
            
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Lyrics with Timestamps", open=False):
                        lyrics_df = gr.Dataframe(label="Lyrics with Timestamps")
                
                with gr.Column():
                    with gr.Accordion("Separated Stems", open=False):
                        stems_html = gr.HTML(label="Separated Audio Stems")
            
            # Connect audio analysis function
            analyze_button.click(
                fn=process_audio,
                inputs=[audio_input],
                outputs=[
                    analysis_output,
                    waveform_plot,
                    spectrogram_plot,
                    chromagram_plot,
                    lyrics_df,
                    sections_df,
                    audio_properties,
                    stems_html
                ]
            )
        
        # Tab 2: Storyboard Generation
        with gr.TabItem("Storyboard", id="storyboard_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    theme_input = gr.Textbox(
                        label="Theme",
                        placeholder="Enter a theme for your music video (e.g., 'cyberpunk future', 'nature journey')",
                        elem_id="theme_input"
                    )
                    
                    with gr.Accordion("Style Preferences (Optional)", open=False):
                        style_preferences = gr.Textbox(
                            label="Style Preferences",
                            placeholder="Enter style preferences, one per line (e.g., 'Art Style: cinematic', 'Color Scheme: vibrant')",
                            lines=5
                        )
                    
                    storyboard_button = gr.Button("Generate Storyboard", variant="primary")
                    storyboard_status = gr.Markdown(label="Storyboard Status")
                
                with gr.Column(scale=2):
                    with gr.Accordion("Storyboard Timeline", open=True):
                        timeline_plot = gr.Plot(label="Storyboard Timeline")
                    
                    storyboard_output = gr.Dataframe(label="Storyboard")
            
            with gr.Row():
                with gr.Column():
                    scene_index = gr.Number(
                        label="Preview Scene (Enter scene number, starting from 0)",
                        value=0,
                        precision=0
                    )
                    preview_button = gr.Button("Preview Scene")
                
                with gr.Column():
                    scene_details = gr.Markdown(label="Scene Details")
            
            with gr.Row():
                with gr.Column():
                    scene_image = gr.Image(label="Scene Preview Image", type="filepath")
                    color_palette = gr.Plot(label="Color Palette")
                
                with gr.Column():
                    scene_video = gr.Video(label="Scene Preview Video")
            
            # Connect storyboard generation function
            storyboard_button.click(
                fn=create_storyboard_from_analysis,
                inputs=[theme_input, style_preferences],
                outputs=[storyboard_status, storyboard_output, timeline_plot]
            )
            
            # Connect scene preview function
            preview_button.click(
                fn=preview_storyboard_scene,
                inputs=[scene_index],
                outputs=[scene_details, scene_image, scene_video, color_palette]
            )
        
        # Tab 3: Visual Generation
        with gr.TabItem("Generation", id="generation_tab"):
            with gr.Row():
                with gr.Column():
                    transition_type = gr.Radio(
                        label="Transition Type",
                        choices=["none", "fade", "slide", "mixed"],
                        value="fade"
                    )
                    generate_button = gr.Button("Generate Music Video", variant="primary", size="lg")
                    generation_status = gr.Markdown(label="Generation Status")
                
                with gr.Column():
                    final_video = gr.Video(label="Generated Music Video")
            
            with gr.Accordion("Advanced Options", open=False):
                gr.Markdown("""
                ### Advanced Generation Options
                
                These options will be implemented in a future update:
                
                - Custom resolution selection
                - Frame rate adjustment
                - Quality presets (draft, standard, high quality)
                - Custom transition durations
                - Export options (MP4, WebM, GIF)
                """)
            
            # Connect music video generation function
            generate_button.click(
                fn=generate_music_video,
                inputs=[transition_type],
                outputs=[generation_status, final_video]
            )
    
    # Sync audio input across tabs
    audio_input.change(
        lambda x: x,
        inputs=[audio_input],
        outputs=[audio_input_ref]
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)
