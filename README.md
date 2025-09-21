# Neural Style Transfer App

Simple web app for applying artistic styles to photos. This is my final year project - sorry if my English is not perfect, Spanish is my first language.

## What it does
- Upload your photo + an art image (style)
- App transfers the artistic style to your photo
- Uses classic Gatys method and AdaIN algorithm
- Web interface made with Flask

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the app
For Windows:
```bash
set FLASK_APP=app
set FLASK_ENV=development
flask run
```

Or simply:
```bash
python -m flask --app app run --debug
```

Open your browser: http://127.0.0.1:5000

## Project structure
- `app/nst/` - Neural style transfer algorithms
- `app/web/` - Web routes and Flask stuff  
- `app/nst/pipeline/` - Main style transfer methods (Gatys, AdaIN)
- `app/nst/utils/` - Image processing helpers
- `static/` `templates/` - Frontend files

## Usage tips
- Use smaller images first (like 512px) - it's faster
- First run takes more time (downloads model weights)
- If you have NVIDIA GPU, it will be much faster
- Clear cache if app uses too much memory

## Methods available
- **Gatys**: Original neural style transfer (slower but good quality)
- **AdaIN**: Faster real-time style transfer

Made for university project. Feel free to improve the code if you find bugs.
