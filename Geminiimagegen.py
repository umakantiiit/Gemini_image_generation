
import google
import streamlit as st
import google.generativeai as genai
from google.generativeai import types

from PIL import Image
from io import BytesIO
import base64

# Initialize the Gemini API client with your API key
genai.configure(api_key='AIzaSyDz6Vzwjo2WpvJszB6FuPLQ7Emmvv6QLqc')
model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')
#client = genai.Client(api_key='AIzaSyDz6Vzwjo2WpvJszB6FuPLQ7Emmvv6QLqc')

def generate_image(prompt, uploaded_images):
    """
    Generate image using the Gemini API.
    If images are uploaded, they are added to the API call alongside the prompt.
    """
    # Prepare the contents list for the API call.
    # Always include the prompt first.
    contents = [prompt]
    
    # If the user has uploaded images, add them.
    # Allow up to 2 images.
    for file in uploaded_images:
        try:
            image = Image.open(file)
            contents.append(image)
        except Exception as e:
            st.error(f"Error reading image: {e}")
            return None
    generation_config = genai.types.GenerateContentConfig(response_modalities=['Text', 'Image']) 
    # Make the API call using the Gemini model
    response = model.generate_content(
        #model="gemini-2.0-flash-exp-image-generation",
        contents=contents,
        generation_config=generation_config
    )

    # Process the response: look for the first image in the candidate parts.
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            try:
                # Convert inline data to image
                gen_image = Image.open(BytesIO(part.inline_data.data))
                return gen_image
            except Exception as e:
                st.error(f"Error processing generated image: {e}")
                return None

    st.error("No image was generated. Please try again.")
    return None

def main():
    st.title("Gemini Image Generation App")

    st.write("Upload up to 2 images (optional) and provide a prompt to generate a new image.")

    # File uploader: allow multiple files (up to 2)
    uploaded_files = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files and len(uploaded_files) > 2:
        st.warning("You can upload a maximum of 2 images. Only the first 2 will be used.")
        uploaded_files = uploaded_files[:2]

    # Prompt text input (mandatory)
    prompt = st.text_input("Enter prompt for image generation", help="This field is mandatory.")

    if st.button("Generate Image"):
        if not prompt.strip():
            st.error("Please enter a prompt to generate an image.")
        else:
            st.info("Generating image, please wait...")
            gen_image = generate_image(prompt, uploaded_files if uploaded_files else [])
            if gen_image:
                st.success("Image generated successfully!")
                # Display the generated image
                st.image(gen_image, caption="Generated Image", use_column_width=True)
                # Prepare download button: convert image to bytes
                buf = BytesIO()
                gen_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(label="Download Image", data=byte_im, file_name="gemini-native-image.png", mime="image/png")

if __name__ == '__main__':
    main()
