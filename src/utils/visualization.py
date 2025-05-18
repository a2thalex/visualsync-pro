import gradio as gr
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

def create_audio_visualizations(analysis_results):
    """
    Create visualization components for audio analysis results
    
    Args:
        analysis_results: Dictionary containing audio analysis results
        
    Returns:
        dict: Dictionary of Gradio components for visualization
    """
    components = {}
    
    # Extract visualization data
    visualizations = analysis_results.get("visualizations", {})
    
    # Create waveform component if available
    if "waveform" in visualizations and visualizations["waveform"]:
        try:
            waveform_fig = go.Figure(json.loads(visualizations["waveform"]))
            components["waveform"] = gr.Plot(value=waveform_fig)
        except Exception as e:
            print(f"Error creating waveform component: {str(e)}")
            components["waveform"] = gr.Plot(label="Waveform Visualization")
    
    # Create spectrogram component if available
    if "spectrogram" in visualizations and visualizations["spectrogram"]:
        try:
            spectrogram_fig = go.Figure(json.loads(visualizations["spectrogram"]))
            components["spectrogram"] = gr.Plot(value=spectrogram_fig)
        except Exception as e:
            print(f"Error creating spectrogram component: {str(e)}")
            components["spectrogram"] = gr.Plot(label="Spectrogram Visualization")
    
    # Create chromagram component if available
    if "chromagram" in visualizations and visualizations["chromagram"]:
        try:
            chromagram_fig = go.Figure(json.loads(visualizations["chromagram"]))
            components["chromagram"] = gr.Plot(value=chromagram_fig)
        except Exception as e:
            print(f"Error creating chromagram component: {str(e)}")
            components["chromagram"] = gr.Plot(label="Chromagram Visualization")
    
    return components

def create_lyrics_visualization(lyrics_segments):
    """
    Create visualization for lyrics with timestamps
    
    Args:
        lyrics_segments: List of segments with timestamps
        
    Returns:
        gr.Dataframe: Dataframe component with lyrics and timestamps
    """
    try:
        # Create dataframe for lyrics segments
        lyrics_data = {
            "Start Time": [],
            "End Time": [],
            "Duration": [],
            "Text": []
        }
        
        for segment in lyrics_segments:
            # Extract segment data
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "")
            
            # Add to dataframe
            lyrics_data["Start Time"].append(f"{start:.2f}s")
            lyrics_data["End Time"].append(f"{end:.2f}s")
            lyrics_data["Duration"].append(f"{end - start:.2f}s")
            lyrics_data["Text"].append(text)
        
        return gr.Dataframe(
            value=lyrics_data,
            headers=["Start Time", "End Time", "Duration", "Text"],
            datatype=["str", "str", "str", "str"],
            row_count=(min(len(lyrics_data["Text"]), 10))
        )
    
    except Exception as e:
        print(f"Error creating lyrics visualization: {str(e)}")
        return gr.Dataframe(
            headers=["Start Time", "End Time", "Duration", "Text"],
            datatype=["str", "str", "str", "str"]
        )

def create_sections_visualization(sections):
    """
    Create visualization for audio sections
    
    Args:
        sections: List of section data
        
    Returns:
        gr.Dataframe: Dataframe component with section information
    """
    try:
        # Create dataframe for sections
        sections_data = {
            "Label": [],
            "Start Time": [],
            "End Time": [],
            "Duration": []
        }
        
        for section in sections:
            # Extract section data
            label = section.get("label", "")
            start = section.get("start", 0)
            end = section.get("end", 0)
            duration = section.get("duration", 0)
            
            # Add to dataframe
            sections_data["Label"].append(label)
            sections_data["Start Time"].append(f"{start:.2f}s")
            sections_data["End Time"].append(f"{end:.2f}s")
            sections_data["Duration"].append(f"{duration:.2f}s")
        
        return gr.Dataframe(
            value=sections_data,
            headers=["Label", "Start Time", "End Time", "Duration"],
            datatype=["str", "str", "str", "str"],
            row_count=(min(len(sections_data["Label"]), 10))
        )
    
    except Exception as e:
        print(f"Error creating sections visualization: {str(e)}")
        return gr.Dataframe(
            headers=["Label", "Start Time", "End Time", "Duration"],
            datatype=["str", "str", "str", "str"]
        )

def create_audio_properties_markdown(analysis_results):
    """
    Create markdown component for audio properties
    
    Args:
        analysis_results: Dictionary containing audio analysis results
        
    Returns:
        str: Markdown string with audio properties
    """
    try:
        # Extract properties
        tempo = analysis_results.get("tempo", "Unknown")
        key = analysis_results.get("key", "Unknown")
        duration = analysis_results.get("duration", 0)
        
        # Format as markdown
        markdown = f"""
        ## Audio Properties
        
        | Property | Value |
        | --- | --- |
        | **Tempo** | {tempo:.1f} BPM |
        | **Key** | {key} |
        | **Duration** | {duration:.2f} seconds |
        """
        
        return markdown
    
    except Exception as e:
        print(f"Error creating audio properties markdown: {str(e)}")
        return "Error displaying audio properties"

def create_storyboard_visualization(storyboard):
    """
    Create visualization for storyboard
    
    Args:
        storyboard: List of storyboard scenes
        
    Returns:
        gr.Dataframe: Dataframe component with storyboard information
    """
    try:
        # Create dataframe for storyboard
        storyboard_data = {
            "Time": [],
            "Description": [],
            "Visual Prompt": [],
            "Camera Movement": [],
            "Color Palette": []
        }
        
        for scene in storyboard:
            # Extract scene data
            time = scene.get("time", "")
            description = scene.get("description", "")
            visual_prompt = scene.get("visual_prompt", "")
            camera_movement = scene.get("camera_movement", "")
            color_palette = scene.get("color_palette", "")
            
            # Add to dataframe
            storyboard_data["Time"].append(time)
            storyboard_data["Description"].append(description)
            storyboard_data["Visual Prompt"].append(visual_prompt)
            storyboard_data["Camera Movement"].append(camera_movement)
            storyboard_data["Color Palette"].append(color_palette)
        
        return gr.Dataframe(
            value=storyboard_data,
            headers=["Time", "Description", "Visual Prompt", "Camera Movement", "Color Palette"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=(min(len(storyboard_data["Time"]), 10))
        )
    
    except Exception as e:
        print(f"Error creating storyboard visualization: {str(e)}")
        return gr.Dataframe(
            headers=["Time", "Description", "Visual Prompt", "Camera Movement", "Color Palette"],
            datatype=["str", "str", "str", "str", "str"]
        )

def create_color_palette_visualization(palette):
    """
    Create visualization for color palette
    
    Args:
        palette: List of color hex codes
        
    Returns:
        gr.Plot: Plot component with color palette visualization
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add color swatches
        for i, color in enumerate(palette):
            fig.add_trace(go.Bar(
                x=[i],
                y=[1],
                marker_color=color,
                name=color,
                hoverinfo="name",
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title="Color Palette",
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            height=100,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return gr.Plot(value=fig)
    
    except Exception as e:
        print(f"Error creating color palette visualization: {str(e)}")
        return gr.Plot(label="Color Palette Visualization")

def create_timeline_visualization(storyboard, audio_duration):
    """
    Create timeline visualization for storyboard
    
    Args:
        storyboard: List of storyboard scenes
        audio_duration: Duration of the audio in seconds
        
    Returns:
        gr.Plot: Plot component with timeline visualization
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add timeline
        for i, scene in enumerate(storyboard):
            # Extract scene data
            time_str = scene.get("time", "0s")
            description = scene.get("description", "")
            
            # Convert time to seconds
            if isinstance(time_str, str) and time_str.endswith("s"):
                time = float(time_str[:-1])
            else:
                time = float(time_str)
            
            # Determine end time (either next scene or end of audio)
            if i < len(storyboard) - 1:
                next_time_str = storyboard[i+1].get("time", str(audio_duration) + "s")
                if isinstance(next_time_str, str) and next_time_str.endswith("s"):
                    end_time = float(next_time_str[:-1])
                else:
                    end_time = float(next_time_str)
            else:
                end_time = audio_duration
            
            # Add bar for scene
            fig.add_trace(go.Bar(
                x=[end_time - time],
                y=[0],
                base=[time],
                orientation="h",
                name=f"Scene {i+1}",
                text=description,
                hoverinfo="text",
                marker=dict(
                    color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                )
            ))
        
        # Update layout
        fig.update_layout(
            title="Storyboard Timeline",
            xaxis_title="Time (seconds)",
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            height=150,
            margin=dict(l=20, r=20, t=40, b=20),
            barmode="stack"
        )
        
        return gr.Plot(value=fig)
    
    except Exception as e:
        print(f"Error creating timeline visualization: {str(e)}")
        return gr.Plot(label="Timeline Visualization")
