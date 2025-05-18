# Deploying VisualSync Professional to Hugging Face Spaces

This guide provides detailed information on deploying VisualSync Professional to Hugging Face Spaces, including template options and hardware recommendations.

## Gradio Template Options

When deploying to Hugging Face Spaces, you have two main options:

### Option 1: Use the Existing Configuration (Recommended)

The project already includes a `huggingface-space.yml` file that configures the Space to use Gradio SDK. This is the simplest approach:

1. Create a new Space on Hugging Face
2. Upload the project files (or connect to your Git repository)
3. The Space will automatically use the configuration from `huggingface-space.yml`

### Option 2: Use a Hugging Face Template

If you prefer to start with a template:

1. Create a new Space on Hugging Face
2. Select "Gradio" as the SDK
3. Choose a template (e.g., "Gradio App")
4. Replace the template files with your VisualSync Professional files
5. Make sure to keep the Space's configuration aligned with your project requirements

## Hardware Recommendations

VisualSync Professional performs several computationally intensive tasks:

1. Audio stem separation (CPU-intensive)
2. Image generation with Stable Diffusion XL (GPU-intensive)
3. Video generation with Stable Video Diffusion (GPU-intensive)

### Recommended Hardware Configuration

For optimal performance on Hugging Face Spaces:

#### CPU Requirements
- **Minimum**: 4 vCPUs
- **Recommended**: 8+ vCPUs
- **Optimal**: 16+ vCPUs

Higher CPU core count will significantly improve audio processing speed, especially for stem separation with Demucs.

#### GPU Requirements
- **Minimum**: NVIDIA T4 (16GB VRAM)
- **Recommended**: NVIDIA A10G (24GB VRAM)
- **Optimal**: NVIDIA A100 (40GB VRAM)

The GPU is critical for running the Stable Diffusion models. More VRAM allows for:
- Larger batch sizes
- Higher resolution outputs
- Faster generation times

#### Memory Requirements
- **Minimum**: 16GB RAM
- **Recommended**: 32GB RAM
- **Optimal**: 64GB RAM

Higher RAM is important for processing longer audio files and handling multiple concurrent users.

### Hugging Face Space Hardware Tiers

Based on the above requirements, here are the recommended Hugging Face Space hardware tiers:

1. **For Testing/Development**: CPU or T4 Small
2. **For Personal Use**: T4 Medium
3. **For Production**: A10G or A100

## Optimizing for Different Hardware

The application can be configured to adapt to different hardware capabilities:

### For Lower-End Hardware
- Reduce the default resolution of generated images
- Limit video length or segment longer videos
- Process audio at a lower sample rate
- Use smaller/faster models for inference

### For Higher-End Hardware
- Increase batch sizes for faster generation
- Enable higher resolution outputs
- Process longer audio files
- Use the full-size models for best quality

## Configuration Adjustments

You can modify the following files to adjust for different hardware:

1. `src/utils/media_generator.py`: Adjust batch sizes, model parameters, and resolution settings
2. `src/utils/audio_analyzer.py`: Configure audio processing parameters
3. `src/app.py`: Adjust UI elements and processing options

## Monitoring Resource Usage

When running on Hugging Face Spaces, monitor your resource usage to ensure optimal performance:

1. Check GPU memory usage during image/video generation
2. Monitor CPU usage during audio processing
3. Track overall memory consumption

If you encounter resource limitations, consider upgrading your Space's hardware tier or optimizing your application's resource usage.
