import requests
import json
import base64
from PIL import Image
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

# === Replace with your own Cerebrium endpoint and API key ===
CEREBRIUM_URL = "https://api.cortex.cerebrium.ai/v4/p-fb887699/mtailer-project-01/handler"  # Replace 'predict' with actual function if different
API_KEY = f"Bearer {os.getenv('CEREBRIUM_API_KEY')}"  # Replace with your actual API key

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def predict(image_path):
    """Call Cerebrium endpoint and return prediction"""
    image_base64 = encode_image_to_base64(image_path)

    headers = {
        'Authorization': API_KEY,
        'Content-Type': 'application/json'
    }

    payload = json.dumps({
    "image": image_base64,
    })


    response = requests.post(CEREBRIUM_URL, headers=headers, data=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction for {os.path.basename(image_path)}: Class ID = {result}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

def run_tests():
    print("🧪 Running tests against Cerebrium endpoint...")

    test_images = {
        "n01440764_tench.jpeg": 0,
        "n01667114_mud_turtle.JPEG": 35
        # Add more test cases here if needed
    }

    for img_name, expected_class in test_images.items():
        print(f"\nTesting: {img_name}")
        image_path = os.path.join(".", img_name)
        image_base64 = encode_image_to_base64(image_path)

        headers = {
            'Authorization': API_KEY,
            'Content-Type': 'application/json'
        }

        payload = json.dumps({
            "image": image_base64
        })

        response = requests.post(CEREBRIUM_URL, headers=headers, data=payload)

        if response.status_code == 200:
            result = response.json()
            predicted = result['class_id']
            print(f"Expected: {expected_class}, Predicted: {predicted}")
            assert predicted == expected_class, f"❌ Mismatch for {img_name}"
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

    print("\n✅ All tests completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image for prediction")
    parser.add_argument("--test", action="store_true", help="Run preset tests")

    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.image:
        predict(args.image)
    else:
        print("Please provide --image <path> or use --test")
