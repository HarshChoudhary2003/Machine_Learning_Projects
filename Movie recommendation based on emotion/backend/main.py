from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI(title="Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Movie(BaseModel):
    title: str
    image: Optional[str] = None
    rating: Optional[str] = None
    year: Optional[str] = None
    url: Optional[str] = None

# Using a comprehensive emotion-to-genre map
EMOTION_MAP = {
    "happy": "comedy",
    "sad": "drama",
    "excited": "action",
    "scared": "horror",
    "angry": "thriller",
    "romantic": "romance",
    "bored": "adventure",
    "curious": "documentary",
    "neutral": "animation"
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def clean_title(title_text):
    # Remove leading numbering like "1. Title"
    return re.sub(r'^\d+\.\s*', '', title_text)

@app.get("/recommend", response_model=List[Movie])
async def recommend_movies(emotion: str):
    genre = EMOTION_MAP.get(emotion.lower())
    if not genre:
        # Fallback using broader search or just return empty
        # Or parse the input if it's a direct genre
        if emotion.lower() in EMOTION_MAP.values():
            genre = emotion.lower()
        else:
            raise HTTPException(status_code=400, detail="Emotion not recognized. Try: happy, sad, excited, scared, angry, romantic, bored, curious, neutral.")
            
    url = f'https://www.imdb.com/search/title/?title_type=feature&genres={genre}&sort=num_votes,desc'
    # Adding sort by popularity/votes for better recommendations
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from IMDb: {str(e)}")

    soup = BeautifulSoup(response.text, 'html.parser')
    movies = []
    
    # Selecting items from the new IMDb layout
    # Currently, items are often in `ipc-metadata-list-summary-item` 
    items = soup.find_all('li', class_='ipc-metadata-list-summary-item')
    
    if items:
        for item in items[:12]: # Limit to 12 top movies
            try:
                # Title
                title_tag = item.find('h3', class_='ipc-title__text')
                raw_title = title_tag.get_text() if title_tag else "Unknown Title"
                title = clean_title(raw_title)
                
                # Image
                img_tag = item.find('img', class_='ipc-image')
                img_url = img_tag.get('src') if img_tag else None
                
                # Metadata (Year, Rating etc. are tricky as classes are obfuscated sometimes)
                # We try to grab the first couple of metadata items (usually year, duration, rating)
                metadata_items = item.select('.dli-title-metadata-item')
                year = metadata_items[0].get_text() if metadata_items else None

                # Rating
                rating_span = item.select_one('.ipc-rating-star--base')
                rating = rating_span.get_text(strip=True).split('(')[0] if rating_span else None 
                
                # Link
                link_tag = item.select_one('a.ipc-title-link-wrapper')
                movie_url = f"https://www.imdb.com{link_tag['href']}" if link_tag else None

                if title: 
                     movies.append(Movie(
                        title=title,
                        image=img_url,
                        year=year,
                        rating=rating,
                        url=movie_url
                    ))
            except Exception as e:
                continue
    else:
        # Fallback to the original logic from notebook if the structured class isn't found
        # This is less detailed but ensures we get something
        print("Using fallback scraping logic")
        title_tags = soup.find_all('a', href=re.compile(r'/title/tt\d+/'))
        seen_titles = set()
        
        for tag in title_tags:
            title = tag.get_text().strip()
            href = tag.get('href')
            
            # Simple heuristic to avoid generic links or empty text
            if title and href and title not in seen_titles and len(title) > 3:
                seen_titles.add(title)
                # Try to find an image near this tag if possible, otherwise None
                # This is hard in unstructured mode
                movies.append(Movie(
                    title=title,
                    url=f"https://www.imdb.com{href}"
                ))
                if len(movies) >= 12:
                    break
    
    return movies

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
