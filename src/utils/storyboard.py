import os
import json
import random
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class StoryboardGenerator:
    """Class for generating AI-powered storyboards"""
    
    def __init__(self):
        """Initialize the storyboard generator"""
        pass
    
    def generate_storyboard(self, audio_analysis, theme="Default theme", style_preferences=None):
        """
        Generate a comprehensive storyboard based on audio analysis and theme
        
        Args:
            audio_analysis: Dictionary containing audio analysis results
            theme: Optional theme for the storyboard
            style_preferences: Optional dictionary of style preferences
            
        Returns:
            list: List of scenes with timestamps, descriptions, visual prompts, etc.
        """
        try:
            # Extract audio analysis data
            lyrics = audio_analysis.get("lyrics", "")
            tempo = audio_analysis.get("tempo", "Unknown")
            key = audio_analysis.get("key", "Unknown")
            duration = audio_analysis.get("duration", 0)
            sections = audio_analysis.get("sections", [])
            
            # Create prompt for OpenAI
            prompt = self._create_storyboard_prompt(
                lyrics, tempo, key, duration, sections, theme, style_preferences
            )
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional creative director specializing in music video storyboarding with expertise in visual storytelling, cinematography, and color theory."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            # Extract and parse the response
            storyboard_text = response.choices[0].message.content
            
            # Extract JSON from the response
            # Find the start and end of the JSON content
            json_start = storyboard_text.find("[")
            json_end = storyboard_text.rfind("]") + 1
            
            if json_start != -1 and json_end != -1:
                json_content = storyboard_text[json_start:json_end]
                storyboard = json.loads(json_content)
            else:
                # Fallback if JSON parsing fails
                storyboard = self._create_fallback_storyboard(duration, sections)
            
            # Enhance storyboard with additional details if needed
            enhanced_storyboard = self._enhance_storyboard(storyboard, audio_analysis)
            
            return enhanced_storyboard
        
        except Exception as e:
            print(f"Error generating storyboard: {str(e)}")
            # Create a fallback storyboard
            return self._create_fallback_storyboard(audio_analysis.get("duration", 180), audio_analysis.get("sections", []))
    
    def _create_storyboard_prompt(self, lyrics, tempo, key, duration, sections, theme, style_preferences):
        """
        Create a detailed prompt for the OpenAI API
        
        Args:
            lyrics: Extracted lyrics
            tempo: Tempo in BPM
            key: Musical key
            duration: Duration in seconds
            sections: List of audio sections
            theme: Theme for the storyboard
            style_preferences: Dictionary of style preferences
            
        Returns:
            str: Detailed prompt for OpenAI
        """
        # Format sections for the prompt
        sections_text = ""
        if sections:
            sections_text = "AUDIO SECTIONS:\n"
            for i, section in enumerate(sections):
                start = section.get("start", 0)
                end = section.get("end", 0)
                sections_text += f"- Section {i+1}: {start:.2f}s to {end:.2f}s (duration: {end-start:.2f}s)\n"
        
        # Format style preferences
        style_text = ""
        if style_preferences:
            style_text = "STYLE PREFERENCES:\n"
            for key, value in style_preferences.items():
                style_text += f"- {key}: {value}\n"
        
        # Create the prompt
        prompt = f"""
        Create a detailed storyboard for a professional music video based on the following:
        
        LYRICS:
        {lyrics}
        
        AUDIO PROPERTIES:
        - Tempo: {tempo} BPM
        - Key: {key}
        - Duration: {duration} seconds
        
        {sections_text}
        
        THEME:
        {theme}
        
        {style_text}
        
        For each scene in the storyboard, provide:
        1. Timestamp (in seconds)
        2. Detailed scene description
        3. Visual prompt for AI image generation (be specific and detailed)
        4. Camera movement suggestion (static, pan, zoom, tracking, etc.)
        5. Color palette suggestion (provide 3-5 hex color codes)
        
        Consider the following in your storyboard:
        - Match the visual mood to the audio characteristics
        - Create a cohesive visual narrative that complements the lyrics
        - Suggest appropriate transitions between scenes
        - Vary the visual style to maintain viewer interest
        - Ensure the pacing matches the tempo and energy of the music
        
        Format the response as a JSON array of objects with keys: "time", "description", "visual_prompt", "camera_movement", "color_palette"
        """
        
        return prompt
    
    def _create_fallback_storyboard(self, duration, sections=None):
        """
        Create a fallback storyboard if the API call fails
        
        Args:
            duration: Duration of the audio in seconds
            sections: Optional list of audio sections
            
        Returns:
            list: List of scenes with timestamps, descriptions, and visual prompts
        """
        # Determine scene count and durations
        if sections and len(sections) > 2:
            # Use audio sections if available
            scene_count = len(sections)
            scene_times = [section.get("start", 0) for section in sections]
        else:
            # Otherwise, create evenly spaced scenes
            scene_count = min(8, max(5, int(duration / 30)))  # 1 scene per 30 seconds, min 5, max 8
            scene_times = [i * (duration / scene_count) for i in range(scene_count)]
        
        # Color palettes for fallback
        color_palettes = [
            ["#1A1A2E", "#16213E", "#0F3460", "#E94560"],
            ["#222831", "#393E46", "#00ADB5", "#EEEEEE"],
            ["#F9ED69", "#F08A5D", "#B83B5E", "#6A2C70"],
            ["#F6F6F6", "#D6E4F0", "#1E56A0", "#163172"],
            ["#45062E", "#9A0F98", "#EA0599", "#FECDE5"]
        ]
        
        # Camera movements for fallback
        camera_movements = [
            "Static shot",
            "Slow pan from left to right",
            "Gentle zoom in",
            "Tracking shot following subject",
            "Slow motion",
            "Drone shot from above",
            "Handheld camera for dynamic feel",
            "Dolly zoom for dramatic effect"
        ]
        
        storyboard = []
        
        for i in range(scene_count):
            time = scene_times[i]
            
            # Select random color palette and camera movement
            palette = random.choice(color_palettes)
            camera = random.choice(camera_movements)
            
            scene = {
                "time": f"{int(time)}s",
                "description": f"Scene {i+1}: Abstract visualization with dynamic elements that match the music's energy at this timestamp.",
                "visual_prompt": f"Abstract digital art with flowing shapes and vibrant colors, high quality, detailed, cinematic lighting",
                "camera_movement": camera,
                "color_palette": palette
            }
            storyboard.append(scene)
        
        return storyboard
    
    def _enhance_storyboard(self, storyboard, audio_analysis):
        """
        Enhance storyboard with additional details based on audio analysis
        
        Args:
            storyboard: Initial storyboard
            audio_analysis: Audio analysis results
            
        Returns:
            list: Enhanced storyboard
        """
        # Extract audio features
        tempo = audio_analysis.get("tempo", 120)
        key = audio_analysis.get("key", "C Major")
        
        # Determine mood based on tempo and key
        mood = self._determine_mood(tempo, key)
        
        # Enhance each scene
        for scene in storyboard:
            # Ensure color palette is a list
            if isinstance(scene.get("color_palette", ""), str):
                # Convert comma-separated string to list
                palette_str = scene.get("color_palette", "")
                if palette_str and "," in palette_str:
                    scene["color_palette"] = [color.strip() for color in palette_str.split(",")]
                else:
                    # Generate a random palette if none provided
                    scene["color_palette"] = self._generate_color_palette(mood)
            
            # Ensure camera_movement is provided
            if not scene.get("camera_movement"):
                scene["camera_movement"] = self._suggest_camera_movement(tempo)
        
        return storyboard
    
    def _determine_mood(self, tempo, key):
        """
        Determine mood based on tempo and key
        
        Args:
            tempo: Tempo in BPM
            key: Musical key
            
        Returns:
            str: Mood description
        """
        # Simplified mood determination
        if "Minor" in key:
            if tempo < 80:
                return "melancholic"
            elif tempo < 120:
                return "mysterious"
            else:
                return "intense"
        else:  # Major key
            if tempo < 80:
                return "peaceful"
            elif tempo < 120:
                return "uplifting"
            else:
                return "energetic"
    
    def _generate_color_palette(self, mood):
        """
        Generate a color palette based on mood
        
        Args:
            mood: Mood description
            
        Returns:
            list: List of hex color codes
        """
        # Predefined palettes based on mood
        palettes = {
            "melancholic": ["#2C3E50", "#34495E", "#95A5A6", "#D6DBDF", "#ECF0F1"],
            "mysterious": ["#1A1A2E", "#16213E", "#0F3460", "#E94560", "#533483"],
            "intense": ["#7D0633", "#B80D57", "#F8615A", "#FFD868", "#1F1F1F"],
            "peaceful": ["#48466D", "#3D84A8", "#46CDCF", "#ABEDD8", "#FFFFFF"],
            "uplifting": ["#F9ED69", "#F08A5D", "#B83B5E", "#6A2C70", "#F5F5F5"],
            "energetic": ["#F7FD04", "#F9B208", "#F98404", "#FC5404", "#FFFFFF"]
        }
        
        return palettes.get(mood, ["#1A1A2E", "#16213E", "#0F3460", "#E94560", "#FFFFFF"])
    
    def _suggest_camera_movement(self, tempo):
        """
        Suggest camera movement based on tempo
        
        Args:
            tempo: Tempo in BPM
            
        Returns:
            str: Camera movement suggestion
        """
        # Suggest camera movement based on tempo
        if tempo < 70:
            movements = [
                "Static shot with minimal movement",
                "Very slow zoom out",
                "Gentle dolly movement",
                "Slow pan across scene"
            ]
        elif tempo < 100:
            movements = [
                "Smooth tracking shot",
                "Steady cam following subject",
                "Medium-paced pan",
                "Gentle crane shot"
            ]
        elif tempo < 130:
            movements = [
                "Dynamic tracking shot",
                "Handheld camera with stabilization",
                "Circular dolly movement",
                "Quick zoom transitions"
            ]
        else:
            movements = [
                "Energetic handheld camera",
                "Quick cuts between angles",
                "Rapid zoom in/out",
                "Whip pans between subjects"
            ]
        
        return random.choice(movements)


# Create a singleton instance
storyboard_generator = StoryboardGenerator()

def generate_storyboard(audio_analysis, theme="Default theme", style_preferences=None):
    """
    Wrapper function to generate a storyboard using the StoryboardGenerator class
    
    Args:
        audio_analysis: Dictionary containing audio analysis results
        theme: Optional theme for the storyboard
        style_preferences: Optional dictionary of style preferences
        
    Returns:
        list: List of scenes with timestamps, descriptions, visual prompts, etc.
    """
    return storyboard_generator.generate_storyboard(audio_analysis, theme, style_preferences)
