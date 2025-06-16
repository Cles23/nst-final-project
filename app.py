import os
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms

from models.gatys import StyleTransfer

# Define the Flask application
app = Flask(__name__)
app.secret_key = 'your_super_secret_key'
UPLOAD_FOLDER = 'static/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_image(image_path, max_size=512):
    image = Image.open(image_path).convert("RGB")

    size = max(image.size)
    if size > max_size:
        scale = max_size / size
        new_size = tuple([int(dim * scale) for dim in image.size])
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0))
    ])

    return transform(image)


@app.route('/')
def index():
    output_image = session.pop('output_image', None)
    return render_template('index.html', output_image=output_image)


@app.route('/stylise', methods=['POST'])
def stylise():
    content_file = request.files['content']
    style_file = request.files['style']

    content_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_file.filename))
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_file.filename))

    content_file.save(content_path)
    style_file.save(style_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = load_image(content_path).to(device)
    style_img = load_image(style_path).to(device)

    model = StyleTransfer(content_img, style_img, device=device)
    output = model.run(steps=300)

    output_image = transforms.ToPILImage()(output.squeeze().cpu())
    output_filename = 'output.jpg'
    output_image.save(os.path.join(app.config['UPLOAD_FOLDER'], output_filename))

    session['output_image'] = output_filename
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)
