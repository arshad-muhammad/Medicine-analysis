from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import google.generativeai as genai
import PIL.Image
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mail import Mail, Message

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure Google Generative AI
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_gemini_response(image_path=None, medicine_name=None):
    """
    Get medicine information using Gemini AI with enhanced prompt
    """
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = '''
    Analyze this image and provide all possible information about any medical or pharmaceutical content visible, regardless of image quality. Even if the image is blurry or partially visible, extract whatever information you can see and provide relevant medical knowledge about similar products. Include:

    1. Basic Information:
       - Any visible text, numbers, or markings
       - If it's a medicine: name (brand/generic), manufacturer, type
       - If it's medical equipment or other items: general category and purpose
       - Physical description (color, shape, size, markings)

    2. Usage & Purpose:
       - Primary medical uses
       - Common applications
       - How it's typically used
       - Typical dosage or usage patterns (if applicable)

    3. Safety Information:
       - Important warnings or precautions
       - Common side effects
       - Who should or shouldn't use it
       - Storage recommendations

    4. Additional Context:
       - Similar products or alternatives
       - General category information
       - Common medical knowledge related to this type of item
       - Relevant medical context

    5. Manufacturing & Composition Details:
       - Detailed composition: list of active and inactive ingredients or materials (if available)
       - Description of the manufacturing process, if visible (e.g., synthetic, natural, biologically derived)
       - Any certifications, approval numbers, or compliance standards (e.g., FDA, CE, ISO)
       - Packaging materials and any special features that might affect product quality or durability
       - Shelf life, expiration date, and any special handling instructions during distribution

    Note: If the image is unclear or lacks specific visible details, focus on identifying what the item might be based on any discernible characteristics, such as shape, color, or markings. Then, provide a general overview of what this type of medical item typically is, does, or includes, even if you can’t verify these exact details from the image alone. For instance: Identify Possible Category: Determine the broad category it might belong to (e.g., tablet, capsule, ointment, syringe, diagnostic tool) based on its general appearance. Provide Relevant Information: Share typical information associated with this category, including general uses, common applications, and related products that often appear similar to this type of item. Describe Standard Medical Knowledge: Include general guidelines, medical knowledge, and warnings that are typically relevant to similar products within this category. For example, if the item appears to be a tablet, you might discuss general storage recommendations, common side effects, and conditions it may be used for, even if these specific details can’t be confirmed from the image alone.

    Important: If some details are unclear because of low image quality, offer general information about similar products or the larger category the item might belong to. Clearly indicate which aspects are inferred rather than directly observed. This approach allows for a broader understanding of the item while being transparent about which details are assumptions based on the item's apparent category or characteristics.
    '''
    
    try:
        if image_path and medicine_name:
            # Both image and text input
            img = PIL.Image.open(image_path)
            response = model.generate_content([prompt, img, f"Additional Information - Medicine Name: {medicine_name}"])
        elif image_path:
            # Only image input
            img = PIL.Image.open(image_path)
            response = model.generate_content([prompt, img])
        elif medicine_name:
            # Only text input
            response = model.generate_content([prompt, f"Medicine Name: {medicine_name}"])
        else:
            return None
        
        return response.text
        
    except Exception as e:
        print(f"Error in Gemini API: {str(e)}")
        return None

def preprocess_image(image):
    """
    Enhanced image preprocessing to handle low quality images
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising with stronger parameters
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Additional sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/why-us')
def about():
    return render_template('about.html') 

@app.route('/contact')
def contact():
    return render_template('contact.html') 

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        medicine_name = request.form.get('medicine_name', '').strip()
        image_file = request.files.get('image')
        processed_path = None
        
        # Process image if provided
        if image_file:
            # Save original image
            filename = secure_filename(image_file.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(original_path)
            
            # Read and preprocess image
            image = cv2.imread(original_path)
            processed_image = preprocess_image(image)
            
            # Save processed image
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
            cv2.imwrite(processed_path, processed_image)
        
        # Get medicine information from Gemini
        medicine_info = get_gemini_response(processed_path, medicine_name)
        
        if medicine_info:
            return jsonify({
                "success": True,
                "data": medicine_info
            })
        else:
            return jsonify({
                "success": False,
                "error": "Could not analyze medicine. Please try again."
            })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)