from google_images_search import GoogleImagesSearch

class ReverseImageSearch:
    def __init__(self, api_key, search_engine_id):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.gis = GoogleImagesSearch(api_key, search_engine_id)

    def search_by_image(self, image_path):
        search_params = {
            'q': '',
            'img': image_path,
        }

        self.gis.search(search_params=search_params)
        results = self.gis.results()
        return results

    def get_first_result_url(self, results):
        if results and len(results) > 0:  # Check if results exist and are not empty
            return results[0].url  # Returning the URL of the first result
        else:
            return "No results found"

