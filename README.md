# VisualSync Professional

A professional-grade music video generation tool using Gradio, OpenAI APIs, and modern AI image/video generation models.

## Features

- **Advanced Audio Analysis**
  - High-quality vocal/instrumental separation using Demucs
  - Comprehensive audio analysis (tempo, key, beat positions, sections)
  - Waveform and spectrogram visualizations
  - Automated lyrics extraction and synchronization with Whisper API

- **Professional UI**
  - Modern, three-tab interface (Analysis, Storyboard, Generation)
  - Professional color palette with Gradio's theming system
  - Clear workflow progression between tabs
  - Expandable/collapsible sections for detailed information
  - Progress indicators for long-running operations

- **AI-Powered Storyboarding**
  - Scene descriptions based on audio characteristics and sections
  - Visual style suggestions that match music mood
  - Camera movement and transition recommendations
  - Color palette suggestions per scene
  - Custom themes and style inputs

- **High-Quality Visual Generation**
  - Stable Diffusion XL for image generation
  - Stable Video Diffusion for motion
  - Scene transitions that match musical changes
  - Visual synchronization to specific audio stems
  - Preview generation for individual scenes

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/visualsync-pro.git
cd visualsync-pro

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys
# Edit the .env file and add your API keys
```

## Usage

```bash
# Run the application
python src/app.py
```

Access the application at http://localhost:7860

## Workflow

1. **Analysis Tab**: Upload your audio file for analysis and stem separation
2. **Storyboard Tab**: Generate and customize your music video storyboard
3. **Generation Tab**: Preview individual scenes and generate the complete music video

## Deployment

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Push this repository to your Space
3. Set up the required environment variables (API keys)

### Docker

```bash
# Build the Docker image
docker build -t visualsync-pro .

# Run the container
docker run -p 7860:7860 visualsync-pro
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that is short and to the point. It lets people do anything with your code with proper attribution and without warranty.

## Acknowledgements

- [Gradio](https://www.gradio.app/) for the web interface
- [OpenAI](https://openai.com/) for GPT and Whisper APIs
- [Demucs](https://github.com/facebookresearch/demucs) for audio source separation
- [Hugging Face](https://huggingface.co/) for Stable Diffusion models
