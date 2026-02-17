from PIL import Image, ImageDraw, ImageFont

def create_placeholder_image(output_path="ui.png", width=800, height=400, color=(30, 30, 46)):
    image = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(image)
    
    # Draw some placeholder UI structure
    draw.rectangle([50, 50, 750, 100], fill=(50, 50, 80), outline=(100, 100, 150))
    draw.text((60, 60), "IPL Predictor AI - UI Preview", fill=(255, 255, 255))
    
    draw.rectangle([50, 120, 250, 180], fill=(30, 30, 46), outline=(100, 100, 150))
    draw.text((60, 130), "Venue Input", fill=(200, 200, 200))
    
    draw.rectangle([300, 120, 500, 180], fill=(30, 30, 46), outline=(100, 100, 150))
    draw.text((310, 130), "Team Input", fill=(200, 200, 200))
    
    draw.rectangle([550, 120, 750, 350], fill=(40, 40, 60), outline=(0, 242, 96))
    draw.text((570, 150), "Prediction Result", fill=(0, 242, 96))
    draw.text((600, 200), "175", fill=(255, 255, 255))
    
    image.save(output_path)
    print(f"Created placeholder UI image at {output_path}")

if __name__ == "__main__":
    create_placeholder_image()
