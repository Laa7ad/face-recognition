from google_images_search import GoogleImagesSearch

def reverse_image_search(image_path):
    # Set up Google Images Search API credentials
    gis = GoogleImagesSearch('YOUR_GOOGLE_API_KEY', 'YOUR_GOOGLE_CX')

    # Perform reverse image search
    _search_params = {
        'q': 'image',
        'imgType': 'photo',
        'fileType': 'jpg',  # You can adjust the file type based on your needs
    }

    try:
        gis.search(search_params=_search_params, path_to_dir='path/to/save/images')
        results = gis.results()
        return results
    except Exception as e:
        print(f"Error performing reverse image search: {str(e)}")
        return []


