import os
import json
import tempfile
import numpy as np
import librosa
import torch
import torchaudio
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define paths
ASSETS_DIR = Path("assets")
STEMS_DIR = ASSETS_DIR / "generated" / "stems"

# Ensure directories exist
STEMS_DIR.mkdir(parents=True, exist_ok=True)

class AudioAnalyzer:
    """Class for audio analysis and stem separation"""
    
    def __init__(self):
        """Initialize the audio analyzer"""
        # Load Demucs model for stem separation
        self.demucs_model = get_model("htdemucs")
        self.demucs_model.eval()
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.demucs_model.to(self.device)
        
        print(f"Loaded Demucs model on {self.device}")
    
    def analyze_audio(self, audio_path):
        """
        Comprehensive audio analysis including stem separation
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            dict: Dictionary containing analysis results
        """
        try:
            # Extract stems
            stems = self.separate_stems(audio_path)
            
            # Extract lyrics using Whisper
            lyrics_data = self.extract_lyrics_with_timestamps(audio_path)
            
            # Analyze audio features
            features = self.analyze_audio_features(audio_path)
            
            # Generate visualizations
            visualizations = self.generate_visualizations(audio_path, features)
            
            # Combine results
            results = {
                "stems": stems,
                "lyrics": lyrics_data["text"],
                "lyrics_segments": lyrics_data["segments"],
                "tempo": features["tempo"],
                "key": features["key"],
                "beats": features["beats"],
                "sections": features["sections"],
                "duration": features["duration"],
                "visualizations": visualizations
            }
            
            return results
        
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            # Return basic information in case of error
            return {
                "stems": {},
                "lyrics": "Lyrics extraction failed",
                "lyrics_segments": [],
                "tempo": 120,
                "key": "C Major",
                "beats": [],
                "sections": [],
                "duration": 0,
                "visualizations": {}
            }
    
    def separate_stems(self, audio_path):
        """
        Separate audio into stems using Demucs
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            dict: Dictionary of stem paths
        """
        try:
            print(f"Separating stems for {audio_path}")
            
            # Create unique directory for this audio file's stems
            audio_filename = Path(audio_path).stem
            stems_subdir = STEMS_DIR / audio_filename
            stems_subdir.mkdir(exist_ok=True)
            
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            
            # Convert to mono if needed for processing
            if audio.shape[0] > 1:
                audio_mono = torch.mean(audio, dim=0, keepdim=True)
            else:
                audio_mono = audio
            
            # Move to device
            audio = audio.to(self.device)
            
            # Apply Demucs model
            with torch.no_grad():
                sources = apply_model(self.demucs_model, audio, device=self.device)
            
            # Get source names from model
            source_names = self.demucs_model.sources
            
            # Save each stem
            stem_paths = {}
            for i, source in enumerate(source_names):
                source_path = stems_subdir / f"{source}.wav"
                source_audio = sources[i]
                
                # Move back to CPU for saving
                source_audio = source_audio.cpu()
                
                # Save audio
                torchaudio.save(
                    source_path,
                    source_audio,
                    sr
                )
                
                stem_paths[source] = str(source_path)
            
            return stem_paths
        
        except Exception as e:
            print(f"Error separating stems: {str(e)}")
            return {}
    
    def extract_lyrics_with_timestamps(self, audio_path):
        """
        Extract lyrics with timestamps using OpenAI Whisper API
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            dict: Dictionary containing transcribed text and segments with timestamps
        """
        try:
            print(f"Extracting lyrics for {audio_path}")
            
            # Use OpenAI Whisper API for transcription with timestamps
            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
            
            # Extract text and segments
            if hasattr(response, 'text'):
                text = response.text
                segments = response.segments
            else:
                # Parse the JSON response if it's returned as a string
                response_data = json.loads(str(response))
                text = response_data.get("text", "")
                segments = response_data.get("segments", [])
            
            return {
                "text": text,
                "segments": segments
            }
        
        except Exception as e:
            print(f"Error extracting lyrics: {str(e)}")
            return {
                "text": "Lyrics extraction failed. Please try again.",
                "segments": []
            }
    
    def analyze_audio_features(self, audio_path):
        """
        Analyze audio features using librosa
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            dict: Dictionary containing audio features
        """
        try:
            print(f"Analyzing audio features for {audio_path}")
            
            # Load audio with librosa
            y, sr = librosa.load(audio_path)
            
            # Get duration
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Detect tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Extract beats
            _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # Estimate key
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            chroma_sum = np.sum(chroma, axis=1)
            key_index = np.argmax(chroma_sum)
            key = key_names[key_index]
            
            # Determine if major or minor
            minor_profile = np.roll([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], 3)
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            
            # Compute correlation with profiles
            major_corr = np.corrcoef(chroma_sum, major_profile)[0, 1]
            minor_corr = np.corrcoef(chroma_sum, minor_profile)[0, 1]
            
            # Determine mode
            mode = "Major" if major_corr >= minor_corr else "Minor"
            key_with_mode = f"{key} {mode}"
            
            # Segment the audio
            segments = librosa.segment.agglomerative(chroma, 8)
            segment_times = librosa.frames_to_time(segments, sr=sr)
            
            # Format sections
            sections = []
            for i in range(len(segment_times) - 1):
                start_time = segment_times[i]
                end_time = segment_times[i+1]
                sections.append({
                    "start": float(start_time),
                    "end": float(end_time),
                    "duration": float(end_time - start_time),
                    "label": f"Section {i+1}"
                })
            
            return {
                "tempo": float(tempo),
                "key": key_with_mode,
                "beats": beat_times.tolist(),
                "sections": sections,
                "duration": float(duration)
            }
        
        except Exception as e:
            print(f"Error analyzing audio features: {str(e)}")
            return {
                "tempo": 120.0,
                "key": "C Major",
                "beats": [],
                "sections": [],
                "duration": 0.0
            }
    
    def generate_visualizations(self, audio_path, features=None):
        """
        Generate waveform and spectrogram visualizations
        
        Args:
            audio_path: Path to the audio file
            features: Optional pre-computed features
            
        Returns:
            dict: Dictionary containing visualization data
        """
        try:
            print(f"Generating visualizations for {audio_path}")
            
            # Load audio if features not provided
            y, sr = librosa.load(audio_path)
            
            # Generate waveform
            waveform_fig = self._create_waveform_plot(y, sr, features)
            
            # Generate spectrogram
            spectrogram_fig = self._create_spectrogram_plot(y, sr, features)
            
            # Generate chromagram
            chromagram_fig = self._create_chromagram_plot(y, sr)
            
            return {
                "waveform": waveform_fig,
                "spectrogram": spectrogram_fig,
                "chromagram": chromagram_fig
            }
        
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            return {}
    
    def _create_waveform_plot(self, y, sr, features=None):
        """Create interactive waveform plot with Plotly"""
        try:
            # Create time axis
            time = np.linspace(0, len(y) / sr, len(y))
            
            # Create figure
            fig = go.Figure()
            
            # Add waveform
            fig.add_trace(go.Scatter(
                x=time,
                y=y,
                mode='lines',
                name='Waveform',
                line=dict(color='rgba(31, 119, 180, 0.8)', width=1)
            ))
            
            # Add beats if available
            if features and 'beats' in features and features['beats']:
                beat_y = np.zeros_like(features['beats'])
                fig.add_trace(go.Scatter(
                    x=features['beats'],
                    y=beat_y,
                    mode='markers',
                    name='Beats',
                    marker=dict(color='red', size=8, symbol='line-ns')
                ))
            
            # Add sections if available
            if features and 'sections' in features and features['sections']:
                for i, section in enumerate(features['sections']):
                    fig.add_shape(
                        type="line",
                        x0=section['start'],
                        y0=-1,
                        x1=section['start'],
                        y1=1,
                        line=dict(color="green", width=2, dash="dash"),
                    )
                    fig.add_annotation(
                        x=section['start'] + section['duration']/2,
                        y=0.9,
                        text=section['label'],
                        showarrow=False,
                        bgcolor="rgba(0,255,0,0.3)"
                    )
            
            # Update layout
            fig.update_layout(
                title="Audio Waveform",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                template="plotly_white",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig.to_json()
        
        except Exception as e:
            print(f"Error creating waveform plot: {str(e)}")
            return None
    
    def _create_spectrogram_plot(self, y, sr, features=None):
        """Create interactive spectrogram plot with Plotly"""
        try:
            # Compute spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            
            # Create figure
            fig = go.Figure()
            
            # Add spectrogram as heatmap
            fig.add_trace(go.Heatmap(
                z=D,
                colorscale='Viridis',
                showscale=False
            ))
            
            # Update layout
            fig.update_layout(
                title="Spectrogram",
                yaxis_title="Frequency",
                xaxis_title="Time",
                template="plotly_white",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            # Update y-axis to show frequency in Hz
            fig.update_yaxes(
                tickvals=np.linspace(0, D.shape[0], 5),
                ticktext=[f"{int(sr * i / (2 * D.shape[0]))} Hz" for i in np.linspace(0, D.shape[0], 5)]
            )
            
            # Update x-axis to show time in seconds
            fig.update_xaxes(
                tickvals=np.linspace(0, D.shape[1], 10),
                ticktext=[f"{i:.1f}s" for i in np.linspace(0, len(y)/sr, 10)]
            )
            
            return fig.to_json()
        
        except Exception as e:
            print(f"Error creating spectrogram plot: {str(e)}")
            return None
    
    def _create_chromagram_plot(self, y, sr):
        """Create interactive chromagram plot with Plotly"""
        try:
            # Compute chromagram
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Create figure
            fig = go.Figure()
            
            # Add chromagram as heatmap
            fig.add_trace(go.Heatmap(
                z=chroma,
                colorscale='Viridis',
                showscale=False
            ))
            
            # Update layout
            fig.update_layout(
                title="Chromagram (Harmonic Content)",
                yaxis_title="Pitch Class",
                xaxis_title="Time",
                template="plotly_white",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            # Update y-axis to show pitch classes
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            fig.update_yaxes(
                tickvals=np.arange(12),
                ticktext=pitch_classes
            )
            
            # Update x-axis to show time in seconds
            fig.update_xaxes(
                tickvals=np.linspace(0, chroma.shape[1], 10),
                ticktext=[f"{i:.1f}s" for i in np.linspace(0, len(y)/sr, 10)]
            )
            
            return fig.to_json()
        
        except Exception as e:
            print(f"Error creating chromagram plot: {str(e)}")
            return None


# Create a singleton instance
audio_analyzer = AudioAnalyzer()

def analyze_audio(audio_path):
    """
    Wrapper function to analyze audio using the AudioAnalyzer class
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        dict: Dictionary containing analysis results
    """
    return audio_analyzer.analyze_audio(audio_path)
