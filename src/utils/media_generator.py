import os
import time
import json
import torch
import numpy as np
from pathlib import Path
import moviepy.editor as mp
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Hugging Face API
hf_token = os.getenv("HUGGINGFACE_TOKEN")
hf_api = HfApi(token=hf_token)

# Define paths
ASSETS_DIR = Path("assets")
GENERATED_IMAGES_DIR = ASSETS_DIR / "generated" / "images"
GENERATED_VIDEOS_DIR = ASSETS_DIR / "generated" / "videos"

# Ensure directories exist
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

class MediaGenerator:
    """Class for generating images and videos using AI models"""
    
    def __init__(self):
        """Initialize the media generator"""
        # Initialize with None, will load on demand to save memory
        self.sdxl_pipeline = None
        self.svd_pipeline = None
        
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Media generator initialized with device: {self.device}")
    
    def _load_sdxl_pipeline(self):
        """Load the Stable Diffusion XL pipeline if not already loaded"""
        if self.sdxl_pipeline is None:
            try:
                print("Loading Stable Diffusion XL pipeline...")
                
                # Load SDXL pipeline
                self.sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.device.type == "cuda" else None,
                    token=hf_token
                )
                
                # Move to device
                self.sdxl_pipeline = self.sdxl_pipeline.to(self.device)
                
                # Use DPMSolver for faster inference
                self.sdxl_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.sdxl_pipeline.scheduler.config
                )
                
                # Enable memory optimization if on GPU
                if self.device.type == "cuda":
                    self.sdxl_pipeline.enable_attention_slicing()
                    self.sdxl_pipeline.enable_vae_slicing()
                
                print("SDXL pipeline loaded successfully")
            
            except Exception as e:
                print(f"Error loading SDXL pipeline: {str(e)}")
                self.sdxl_pipeline = None
    
    def _load_svd_pipeline(self):
        """Load the Stable Video Diffusion pipeline if not already loaded"""
        if self.svd_pipeline is None:
            try:
                print("Loading Stable Video Diffusion pipeline...")
                
                # Load SVD pipeline
                self.svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    token=hf_token
                )
                
                # Move to device
                self.svd_pipeline = self.svd_pipeline.to(self.device)
                
                # Use Euler scheduler
                self.svd_pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    self.svd_pipeline.scheduler.config
                )
                
                # Enable memory optimization if on GPU
                if self.device.type == "cuda":
                    self.svd_pipeline.enable_attention_slicing()
                
                print("SVD pipeline loaded successfully")
            
            except Exception as e:
                print(f"Error loading SVD pipeline: {str(e)}")
                self.svd_pipeline = None
    
    def generate_images(self, prompts, negative_prompt=None, style_params=None, batch_size=1):
        """
        Generate images based on prompts using Stable Diffusion XL
        
        Args:
            prompts: List of prompts or comma-separated string of prompts
            negative_prompt: Optional negative prompt to guide generation
            style_params: Optional dictionary of style parameters
            batch_size: Number of images to generate per prompt
            
        Returns:
            list: List of paths to generated images
        """
        # Convert comma-separated string to list if needed
        if isinstance(prompts, str):
            prompt_list = [p.strip() for p in prompts.split(",")]
        else:
            prompt_list = prompts
        
        # Ensure we have at least one prompt
        if not prompt_list or (len(prompt_list) == 1 and not prompt_list[0]):
            prompt_list = ["Abstract colorful visualization, digital art, high quality"]
        
        # Set default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = "blurry, bad quality, distorted, low resolution, ugly, pixelated"
        
        # Set default style parameters if not provided
        if style_params is None:
            style_params = {
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 768
            }
        
        image_paths = []
        
        try:
            # Load SDXL pipeline if not already loaded
            self._load_sdxl_pipeline()
            
            if self.sdxl_pipeline is None:
                raise Exception("Failed to load SDXL pipeline")
            
            # Process each prompt
            for i, prompt in enumerate(prompt_list):
                print(f"Generating image for prompt: {prompt}")
                
                # Enhance prompt for better results
                enhanced_prompt = self._enhance_prompt(prompt)
                
                # Generate images
                for j in range(batch_size):
                    # Generate a unique filename
                    timestamp = int(time.time())
                    image_filename = f"image_{timestamp}_{i}_{j}.png"
                    image_path = GENERATED_IMAGES_DIR / image_filename
                    
                    # Generate image
                    with torch.no_grad():
                        image = self.sdxl_pipeline(
                            prompt=enhanced_prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=style_params.get("num_inference_steps", 30),
                            guidance_scale=style_params.get("guidance_scale", 7.5),
                            width=style_params.get("width", 1024),
                            height=style_params.get("height", 768)
                        ).images[0]
                    
                    # Save the image
                    image.save(image_path)
                    
                    # Add the path to the list
                    image_paths.append(str(image_path))
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.5)
        
        except Exception as e:
            print(f"Error generating images: {str(e)}")
            # Create a fallback image
            fallback_path = GENERATED_IMAGES_DIR / "fallback.png"
            self._create_placeholder_image(fallback_path)
            image_paths = [str(fallback_path)]
        
        return image_paths
    
    def generate_videos_from_images(self, image_paths, duration_per_video=3.0, fps=24, motion_params=None):
        """
        Generate videos from images using Stable Video Diffusion
        
        Args:
            image_paths: List of paths to images
            duration_per_video: Duration of each video in seconds
            fps: Frames per second
            motion_params: Optional dictionary of motion parameters
            
        Returns:
            list: List of paths to generated videos
        """
        video_paths = []
        
        try:
            # Load SVD pipeline if not already loaded
            self._load_svd_pipeline()
            
            if self.svd_pipeline is None:
                raise Exception("Failed to load SVD pipeline")
            
            # Set default motion parameters if not provided
            if motion_params is None:
                motion_params = {
                    "num_inference_steps": 25,
                    "min_guidance_scale": 1.0,
                    "max_guidance_scale": 3.0,
                    "fps": fps,
                    "num_frames": int(duration_per_video * fps)
                }
            
            # Process each image
            for i, image_path in enumerate(image_paths):
                print(f"Generating video for image: {image_path}")
                
                # Generate a unique filename
                timestamp = int(time.time())
                video_filename = f"video_{timestamp}_{i}.mp4"
                video_path = GENERATED_VIDEOS_DIR / video_filename
                
                # Load image
                image = Image.open(image_path)
                
                # Resize image if needed (SVD requires specific dimensions)
                width, height = image.size
                if width != height:
                    # Crop to square
                    min_dim = min(width, height)
                    left = (width - min_dim) // 2
                    top = (height - min_dim) // 2
                    right = left + min_dim
                    bottom = top + min_dim
                    image = image.crop((left, top, right, bottom))
                
                # Resize to required dimensions
                image = image.resize((576, 576))
                
                # Generate video frames
                with torch.no_grad():
                    frames = self.svd_pipeline(
                        image,
                        num_inference_steps=motion_params.get("num_inference_steps", 25),
                        min_guidance_scale=motion_params.get("min_guidance_scale", 1.0),
                        max_guidance_scale=motion_params.get("max_guidance_scale", 3.0),
                        num_frames=motion_params.get("num_frames", int(duration_per_video * fps)),
                        fps=motion_params.get("fps", fps)
                    ).frames[0]
                
                # Convert frames to video
                frames_array = np.array(frames)
                
                # Use MoviePy to create video
                clip = mp.ImageSequenceClip(list(frames_array), fps=fps)
                clip.write_videofile(str(video_path), codec="libx264", audio=False)
                
                # Add the path to the list
                video_paths.append(str(video_path))
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
        
        except Exception as e:
            print(f"Error generating videos: {str(e)}")
            # Create a fallback video
            fallback_path = GENERATED_VIDEOS_DIR / "fallback.mp4"
            self._create_placeholder_video(fallback_path)
            video_paths = [str(fallback_path)]
        
        return video_paths
    
    def create_music_video(self, scene_videos, audio_path, transitions=None):
        """
        Create a complete music video from scene videos and audio
        
        Args:
            scene_videos: List of paths to scene videos
            audio_path: Path to audio file
            transitions: Optional list of transition types
            
        Returns:
            str: Path to the generated music video
        """
        try:
            # Generate a unique filename for the video
            timestamp = int(time.time())
            video_filename = f"music_video_{timestamp}.mp4"
            video_path = GENERATED_VIDEOS_DIR / video_filename
            
            print(f"Creating music video from {len(scene_videos)} videos and audio")
            
            # Load audio
            audio_clip = mp.AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            
            # Load video clips
            video_clips = []
            for video_path in scene_videos:
                clip = mp.VideoFileClip(video_path)
                video_clips.append(clip)
            
            # Apply transitions if provided
            if transitions and len(transitions) > 0:
                final_clips = []
                for i, clip in enumerate(video_clips):
                    final_clips.append(clip)
                    
                    # Add transition if not the last clip
                    if i < len(video_clips) - 1:
                        transition_type = transitions[i % len(transitions)]
                        next_clip = video_clips[i + 1]
                        
                        # Apply transition based on type
                        if transition_type == "fade":
                            # Add fade transition
                            transition_duration = min(1.0, clip.duration / 4, next_clip.duration / 4)
                            clip = clip.crossfadeout(transition_duration)
                            next_clip = next_clip.crossfadein(transition_duration)
                        
                        elif transition_type == "slide":
                            # Add slide transition
                            transition_duration = min(1.0, clip.duration / 4, next_clip.duration / 4)
                            # This is a simplified version, in a real implementation you would use CompositeVideoClip
                            # with proper slide animation
                        
                        # Add more transition types as needed
                
                # Use the clips with transitions
                video_clips = final_clips
            
            # Concatenate video clips
            final_video = mp.concatenate_videoclips(video_clips)
            
            # Resize video if needed
            final_video = final_video.resize(width=1920, height=1080)
            
            # Set audio
            final_video = final_video.set_audio(audio_clip)
            
            # Write video file
            final_video.write_videofile(
                str(video_path),
                codec="libx264",
                audio_codec="aac",
                fps=24
            )
            
            return str(video_path)
        
        except Exception as e:
            print(f"Error creating music video: {str(e)}")
            # Create a fallback video
            fallback_path = GENERATED_VIDEOS_DIR / "fallback_music_video.mp4"
            self._create_placeholder_video(fallback_path)
            return str(fallback_path)
    
    def preview_scene(self, prompt, audio_segment_path=None, style_params=None, motion_params=None):
        """
        Generate a preview for a single scene
        
        Args:
            prompt: Visual prompt for the scene
            audio_segment_path: Optional path to audio segment for the scene
            style_params: Optional dictionary of style parameters
            motion_params: Optional dictionary of motion parameters
            
        Returns:
            dict: Dictionary with paths to generated image and video
        """
        try:
            # Generate image
            image_paths = self.generate_images([prompt], style_params=style_params)
            
            if not image_paths:
                raise Exception("Failed to generate image")
            
            # Generate video from image
            video_paths = self.generate_videos_from_images(image_paths, motion_params=motion_params)
            
            if not video_paths:
                raise Exception("Failed to generate video")
            
            # Add audio if provided
            if audio_segment_path:
                # Generate a unique filename for the video
                timestamp = int(time.time())
                preview_filename = f"preview_{timestamp}.mp4"
                preview_path = GENERATED_VIDEOS_DIR / preview_filename
                
                # Load video and audio
                video_clip = mp.VideoFileClip(video_paths[0])
                audio_clip = mp.AudioFileClip(audio_segment_path)
                
                # Trim audio to match video duration if needed
                if audio_clip.duration > video_clip.duration:
                    audio_clip = audio_clip.subclip(0, video_clip.duration)
                
                # Set audio
                video_clip = video_clip.set_audio(audio_clip)
                
                # Write video file
                video_clip.write_videofile(
                    str(preview_path),
                    codec="libx264",
                    audio_codec="aac"
                )
                
                # Update video path
                video_paths = [str(preview_path)]
            
            return {
                "image": image_paths[0],
                "video": video_paths[0]
            }
        
        except Exception as e:
            print(f"Error generating preview: {str(e)}")
            # Create fallback files
            fallback_image = GENERATED_IMAGES_DIR / "fallback_preview.png"
            fallback_video = GENERATED_VIDEOS_DIR / "fallback_preview.mp4"
            self._create_placeholder_image(fallback_image)
            self._create_placeholder_video(fallback_video)
            
            return {
                "image": str(fallback_image),
                "video": str(fallback_video)
            }
    
    def _enhance_prompt(self, prompt):
        """
        Enhance prompt for better image generation results
        
        Args:
            prompt: Original prompt
            
        Returns:
            str: Enhanced prompt
        """
        # Add quality boosters if not already present
        quality_boosters = [
            "high quality", "detailed", "sharp focus", "professional", 
            "cinematic lighting", "8k", "photorealistic"
        ]
        
        # Check if any quality boosters are already in the prompt
        has_quality_booster = any(booster.lower() in prompt.lower() for booster in quality_boosters)
        
        if not has_quality_booster:
            # Add quality boosters
            enhanced_prompt = f"{prompt}, high quality, detailed, cinematic lighting"
        else:
            enhanced_prompt = prompt
        
        return enhanced_prompt
    
    def _create_placeholder_image(self, path):
        """
        Create a placeholder image for demonstration or fallback purposes
        
        Args:
            path: Path where the image should be saved
        """
        try:
            # Create a simple gradient image
            width, height = 1024, 768
            image = Image.new("RGB", (width, height))
            
            # Generate a gradient
            for y in range(height):
                for x in range(width):
                    r = int(255 * x / width)
                    g = int(255 * y / height)
                    b = int(255 * (x + y) / (width + height))
                    image.putpixel((x, y), (r, g, b))
            
            # Add text
            # In a real implementation, you would use PIL.ImageDraw to add text
            
            # Save the image
            image.save(path)
            
            print(f"Created placeholder image at {path}")
        
        except Exception as e:
            print(f"Error creating placeholder image: {str(e)}")
            # Create an empty file as last resort
            with open(path, "w") as f:
                f.write("Placeholder for generated image")
    
    def _create_placeholder_video(self, path):
        """
        Create a placeholder video for demonstration or fallback purposes
        
        Args:
            path: Path where the video should be saved
        """
        try:
            # Create a simple gradient image
            width, height = 1024, 768
            image = Image.new("RGB", (width, height))
            
            # Generate a gradient
            for y in range(height):
                for x in range(width):
                    r = int(255 * x / width)
                    g = int(255 * y / height)
                    b = int(255 * (x + y) / (width + height))
                    image.putpixel((x, y), (r, g, b))
            
            # Save the image as a temporary file
            temp_image_path = GENERATED_IMAGES_DIR / "temp_placeholder.png"
            image.save(temp_image_path)
            
            # Create a video from the image
            clip = mp.ImageClip(str(temp_image_path), duration=3)
            clip.write_videofile(str(path), fps=24, codec="libx264", audio=False)
            
            # Remove temporary image
            os.remove(temp_image_path)
            
            print(f"Created placeholder video at {path}")
        
        except Exception as e:
            print(f"Error creating placeholder video: {str(e)}")
            # Create an empty file as last resort
            with open(path, "w") as f:
                f.write("Placeholder for generated video")


# Create a singleton instance
media_generator = MediaGenerator()

def generate_images(prompts, negative_prompt=None, style_params=None, batch_size=1):
    """
    Wrapper function to generate images using the MediaGenerator class
    
    Args:
        prompts: List of prompts or comma-separated string of prompts
        negative_prompt: Optional negative prompt to guide generation
        style_params: Optional dictionary of style parameters
        batch_size: Number of images to generate per prompt
        
    Returns:
        list: List of paths to generated images
    """
    return media_generator.generate_images(prompts, negative_prompt, style_params, batch_size)

def generate_videos_from_images(image_paths, duration_per_video=3.0, fps=24, motion_params=None):
    """
    Wrapper function to generate videos from images using the MediaGenerator class
    
    Args:
        image_paths: List of paths to images
        duration_per_video: Duration of each video in seconds
        fps: Frames per second
        motion_params: Optional dictionary of motion parameters
        
    Returns:
        list: List of paths to generated videos
    """
    return media_generator.generate_videos_from_images(image_paths, duration_per_video, fps, motion_params)

def create_music_video(scene_videos, audio_path, transitions=None):
    """
    Wrapper function to create a music video using the MediaGenerator class
    
    Args:
        scene_videos: List of paths to scene videos
        audio_path: Path to audio file
        transitions: Optional list of transition types
        
    Returns:
        str: Path to the generated music video
    """
    return media_generator.create_music_video(scene_videos, audio_path, transitions)

def preview_scene(prompt, audio_segment_path=None, style_params=None, motion_params=None):
    """
    Wrapper function to preview a scene using the MediaGenerator class
    
    Args:
        prompt: Visual prompt for the scene
        audio_segment_path: Optional path to audio segment for the scene
        style_params: Optional dictionary of style parameters
        motion_params: Optional dictionary of motion parameters
        
    Returns:
        dict: Dictionary with paths to generated image and video
    """
    return media_generator.preview_scene(prompt, audio_segment_path, style_params, motion_params)
