from flask import Flask, render_template, request, redirect, url_for
import os
import requests
from werkzeug.utils import secure_filename
from actual import main

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mpeg', 'webm', 'mov'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                # Process the video with YOLOv5 using main function from actual.py
                query = request.form['query']
                output_folder = 'frames'  # Output folder for frames and video
                video_output_path = main(file_path, output_folder, query)
                if video_output_path:
                    return redirect(url_for('result', video_path=video_output_path))
                else:
                    return "Error processing video."
        
        # Check if a URL was provided
        elif 'url' in request.form:
            url = request.form['url']
            response = requests.get(url)
            if response.status_code == 200:
                filename = os.path.basename(url)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                # Process the video with YOLOv5 using main function from actual.py
                query = request.form['query']
                output_folder = 'frames'  # Output folder for frames and video
                video_output_path = main(file_path, output_folder, query)
                if video_output_path:
                    return redirect(url_for('result', video_path=video_output_path))
                else:
                    return "Error processing video."

    return render_template('index.html')

@app.route('/result')
def result():
    video_path = request.args.get('video_path')
    if video_path:
        print(f"Video path: {video_path}")  # Debug print to check the video path
        return f'''
            <h1>Video processed successfully!</h1>
            <video width="640" height="480" controls>
                <source src="{video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        '''
    else:
        return 'No video path provided.'


if __name__ == '__main__':
    app.run(debug=True)

