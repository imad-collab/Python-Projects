from flask import Flask, render_template, request, jsonify
from modules.detect_ai_text import detect_ai_text

# Initialize Flask application
app = Flask(__name__)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle detection logic
@app.route('/detect', methods=['POST'])
def detect():
    try:
        user_text = request.form.get('text')

        if not user_text:
            return jsonify({'error': 'No text received'}), 400

        detection_result = detect_ai_text(user_text)

        print(f"Detection Result: {detection_result}")
        return render_template('result.html', result=detection_result)

    except Exception as e:
        print(f"Backend error in /detect: {str(e)}")
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
